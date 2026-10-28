"""The investigator's fixed, read-only toolkit (docs/ANOMALIES.md).

The investigator agent has no free shell: its one allowed command prefix is
`uv run qlab anomaly`, and the commands under it are these functions. Each answers one
question about a stored trial with numbers, and each is **read-only by construction**: it
re-runs a stored spec on a stored snapshot (CPU only: no network, no agent), prints, and
writes nothing to the registry or to the stored data. The only file a call may write is the
log of the investigation it belongs to (`QLAB_ANOMALY_LOG`), which `qlab.anomalies.investigator`
reads back to check the dossier's evidence against what the tools really printed.

* `contributions` -- which coin on which day earned what: weight, side, price move, funding,
  cost, net contribution, the day's traded dollar volume;
* `coin` -- one coin: listing and last-price facts, whether Hyperliquid lists it, and its daily
  price, volume and trading status over a window;
* `counterfactual` -- the same run with ONE declared rule applied (positions on thin coin-days
  removed; positions near a delisting removed; any one instrument capped; only the other
  venue's listed coins) and the result beside the original: yearly return, worst fall, the
  year table;
* `compare_venues` -- an idea's two stages over their common days: months side by side, the
  coins that earned in each, and whether the other venue lists them;
* `flags` -- the detector's stored flags for the trial.

Every function returns a `Output`: plain text lines for the agent and the CLI, plus the same
numbers as a mapping for tests and `--json`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from qlab.anomalies.rerun import (
    Rule,
    StoredRun,
    apply_rule,
    capped_leg_result,
    daily,
    decompose,
    load_run,
    reselect,
)
from qlab.harness.metrics import compute_metrics, year_breakdown
from qlab.harness.run import RunResult
from qlab.harness.strategy import ZERO_WEIGHT_TOL

LOG_ENV = "QLAB_ANOMALY_LOG"
"""If set, every toolkit call appends `{cmd, output}` as one JSON line to this file."""
MAX_ROWS = 60
"""The most table rows one call prints: an answer the agent cannot read whole is not evidence."""


@dataclass
class Output:
    lines: list[str]
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def log_call(command: str, output: Output, path: str | None = None) -> None:
    """Append one call to the investigation's log, when the environment names one. The log is
    a plain file in `data/anomalies/calls/`, never the registry."""
    target = path or os.environ.get(LOG_ENV)
    if not target:
        return
    file = Path(target)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": " ".join(command.split()), "output": output.text,
                             "at": datetime.now(UTC).isoformat(timespec="seconds")},
                            ensure_ascii=False) + "\n")


def _pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value * 100:+.{digits}f}%"


def _money(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return f"${value / size:,.1f}{unit}"
    return f"${value:,.0f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    out = ["  ".join(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
                     for i, h in enumerate(header))]
    out += ["  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                      for i, c in enumerate(r)) for r in rows]
    return out


def parse_days(spec: str | None) -> list[pd.Timestamp] | None:
    """`2025-11-02,2026-01-30` -> UTC days. None for no filter."""
    if not spec:
        return None
    return [pd.Timestamp(part.strip(), tz="UTC").normalize() for part in spec.split(",")
            if part.strip()]


# --------------------------------------------------------------------------
# contributions


def contributions(session: Session, trial_id: int, *, days: list[pd.Timestamp] | None = None,
                  start: str | None = None, end: str | None = None, instrument: str | None = None,
                  top: int = 20, run: StoredRun | None = None) -> Output:
    """Who earned what: per day and instrument, the weight, side, price move, funding, cost and
    net contribution (all in shares of the strategy's capital), the instrument's traded dollar
    volume that day, and the days' book returns above the table. Rows are the largest by
    absolute contribution; narrow with `days`, `start`/`end`, `instrument`."""
    stored = run or load_run(session, trial_id)
    dec = decompose(stored)
    net, price, funding, cost = (daily(f, "sum") for f in (dec.net, dec.price, dec.funding,
                                                           dec.cost))
    weight = daily(dec.weights, "mean")
    # A day's price move is the compounded move of its bars (a daily book: the one bar).
    move = daily(dec.move.fillna(0.0), "compound")
    volume = dec.volume_usd.groupby(pd.DatetimeIndex(dec.volume_usd.index).tz_convert("UTC")
                                    .normalize()).sum(min_count=1)
    book = daily(dec.book_net, "compound")
    positions = daily(dec.positions.astype(float), "mean")

    keep = net.index
    if days is not None:
        keep = keep.intersection(pd.DatetimeIndex(days))
    if start:
        keep = keep[keep >= pd.Timestamp(start, tz="UTC")]
    if end:
        keep = keep[keep <= pd.Timestamp(end, tz="UTC")]
    cols = [instrument] if instrument else list(net.columns)
    missing = [c for c in cols if c not in net.columns]
    if missing:
        return Output([f"instrument {missing[0]!r} was never held by this book in the judged "
                       "rows (it contributed nothing)"], {"rows": []})
    stacked = net.loc[keep, cols].stack()
    stacked = stacked[stacked != 0.0]
    order = stacked.abs().sort_values(ascending=False).head(min(top, MAX_ROWS)).index
    rows, data_rows = [], []
    for day, coin in order:
        w = float(weight.loc[day, coin])
        data_rows.append({
            "day": str(day.date()), "instrument": coin, "weight": w,
            "side": "long" if w > 0 else "short" if w < 0 else "flat",
            "price_move": float(move.loc[day, coin]), "price": float(price.loc[day, coin]),
            "funding": float(funding.loc[day, coin]), "cost": float(cost.loc[day, coin]),
            "net": float(net.loc[day, coin]),
            "volume_usd": None if pd.isna(volume.loc[day, coin]) else float(volume.loc[day, coin])})
        r = data_rows[-1]
        rows.append([r["day"], coin, f"{r['weight']:+.4f}", r["side"], _pct(r["price_move"]),
                     _pct(r["price"], 2), _pct(r["funding"], 3), _pct(r["cost"], 3),
                     _pct(r["net"], 2), _money(r["volume_usd"])])
    shown_days = sorted({d for d, _ in order})
    lines = [f"trial {trial_id} ({stored.idea_id}, {stored.venue}, {stored.interval}); "
             "all contributions are shares of the strategy's capital, summed over the day's bars",
             f"reconciles with the stand's run: {dec.reconciles} (largest gap {dec.max_gap:.2e})",
             f"rows: the {len(rows)} largest of {len(stacked)} nonzero coin-days in the selection",
             "day = the DECISION day: the weight set at its close earns the price move of the "
             "NEXT bar (the 'move' column), so a coin's own price table shows that move on the "
             "following day", ""]
    if shown_days:
        lines.append("book on the days shown (compounded net return, mean positions held):")
        lines += [f"  {d.date()}  {_pct(book.loc[d], 2)}  {positions.loc[d]:.1f} positions"
                  for d in shown_days[:15]]
        lines.append("")
    lines += _table(["day", "instrument", "weight", "side", "move", "price", "funding", "cost",
                     "net", "volume"], rows)
    return Output(lines, {"rows": data_rows, "reconciles": dec.reconciles,
                          "selection_coin_days": int(len(stacked)),
                          "book": {str(d.date()): float(book.loc[d]) for d in shown_days}})


# --------------------------------------------------------------------------
# coin


def listing_panel(session: Session, run: StoredRun):
    """The Hyperliquid panel the investigation reads listings from: the idea's own second
    stage when it has one, else the widest Hyperliquid snapshot in the registry. None when
    the registry holds none."""
    from qlab.data.snapshot import load_snapshot
    from qlab.registry.models import DataSnapshot, Spec, Trial
    from qlab.registry.stages import latest_routed, stage2_child

    child = stage2_child(session, run.idea_id)
    if child is not None:
        found = latest_routed(session, child.id)
        if found is not None and found[0].snapshot_id:
            return load_snapshot(found[0].snapshot_id, session=session)
    own = session.get(Trial, run.trial_id)
    if own is not None:
        spec = session.get(Spec, own.spec_id)
        if spec is not None and (spec.data_requirements or {}).get("source") == "hyperliquid":
            return run.panel
    rows = (session.query(DataSnapshot).filter(DataSnapshot.source == "hyperliquid").all())
    if not rows:
        return None
    best = max(rows, key=lambda r: (len(r.instruments or {}), str(r.range_end)))
    return load_snapshot(best.id, session=session)


def hl_names(run: StoredRun, hl_columns: list[str]) -> dict[str, str | None]:
    """The venue's name for each of the run's perpetual columns (None: not listed)."""
    from qlab.pipeline.stage2 import venue_name

    return {c: venue_name(c, hl_columns) for c in run.panel.prices.columns
            if not c.endswith("-SPOT")}


def coin(session: Session, trial_id: int, instrument: str, *, start: str | None = None,
         end: str | None = None, rows: int = 30, run: StoredRun | None = None) -> Output:
    """One coin in the trial's data: when its price starts and stops, its volume, how the
    stand treated it (tradeable), whether Hyperliquid lists it, and its daily price, dollar
    volume and trading status over `start`..`end` (default: the last `rows` days it traded)."""
    stored = run or load_run(session, trial_id)
    panel = stored.panel
    if instrument not in panel.prices.columns:
        near = [c for c in panel.prices.columns if instrument.lower() in c.lower()][:8]
        return Output([f"{instrument!r} is not in this trial's data; similar: {near}"], {})
    price = panel.prices[instrument]
    first, last = price.first_valid_index(), price.last_valid_index()
    stopped = last is not None and last < panel.prices.index[-1] - pd.Timedelta(days=1)
    usd = (panel.volume[instrument] * price)
    held = (stored.weights[instrument].abs() > ZERO_WEIGHT_TOL)
    hl = listing_panel(session, stored)
    hl_name, hl_info = None, "no Hyperliquid snapshot in the registry"
    if hl is not None:
        hl_name = hl_names(stored, list(hl.prices.columns)).get(instrument)
        if hl_name is None:
            hl_info = "NOT listed on Hyperliquid (no matching column in its snapshot)"
        else:
            hp = hl.prices[hl_name]
            ends = ("runs to the end of the snapshot"
                    if hp.last_valid_index() >= hl.prices.index[-1] - pd.Timedelta(days=1)
                    else f"stopped {hp.last_valid_index().date()}")
            hl_info = (f"listed on Hyperliquid as {hl_name}; its price starts "
                       f"{hp.first_valid_index().date()} and {ends}")
    lines = [
        f"coin {instrument} in trial {trial_id} ({stored.venue}, {stored.interval})",
        f"price from {first.date() if first is not None else '—'} to "
        f"{last.date() if last is not None else '—'}; panel ends {panel.prices.index[-1].date()}",
        "stopped trading before the panel's end" if stopped
        else "still trading at the panel's end",
        f"days held by the book: {int(held.sum())} of {len(held)} rows",
        f"median daily dollar volume: {_money(float(usd.median()))}; "
        f"minimum: {_money(float(usd.min()))}; maximum: {_money(float(usd.max()))}",
        hl_info, ""]
    window = price.dropna().index
    if start:
        window = window[window >= pd.Timestamp(start, tz="UTC")]
    if end:
        window = window[window <= pd.Timestamp(end, tz="UTC")]
    window = window[-min(rows, MAX_ROWS):] if not start else window[:min(rows, MAX_ROWS)]
    table = []
    for ts in window:
        i = panel.prices.index.get_loc(ts)
        own = (price.iloc[i] / price.iloc[i - 1] - 1.0) if i > 0 and pd.notna(price.iloc[i - 1]) \
            else float("nan")
        nxt = (price.iloc[i + 1] / price.iloc[i] - 1.0) if i + 1 < len(price) \
            and pd.notna(price.iloc[i + 1]) else float("nan")
        table.append([str(ts.date()), f"{price.iloc[i]:.6g}", _pct(own), _pct(nxt),
                      _money(usd.iloc[i]),
                      "yes" if bool(panel.tradeable[instrument].iloc[i]) else "no",
                      f"{stored.weights[instrument].iloc[i]:+.4f}"])
    lines += ["'move' is the bar's own price change; 'next move' is the change over the NEXT "
              "bar, which is what the weight set at this bar's close earns", ""]
    lines += _table(["day", "price", "move", "next move", "volume", "tradeable", "weight held"],
                    table)
    return Output(lines, {
        "first_price": None if first is None else str(first.date()),
        "last_price": None if last is None else str(last.date()), "stopped": bool(stopped),
        "days_held": int(held.sum()), "hyperliquid_name": hl_name,
        "median_volume_usd": float(usd.median())})


# --------------------------------------------------------------------------
# counterfactual


def _slice_result(result: RunResult, start: pd.Timestamp | None) -> RunResult:
    if start is None:
        return result
    cut = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)
           if isinstance(getattr(result, f.name), pd.Series)}
    return dataclasses.replace(result, **{k: v.loc[v.index >= start] for k, v in cut.items()})


def summarize_result(result: RunResult) -> dict[str, Any]:
    """Yearly return, total, worst fall, Sharpe, and the year table of one run."""
    metrics = compute_metrics(result)
    years = year_breakdown(result.net_return)
    table = {int(k.split("_")[1]): {"return": v, "days": years.get(f"year_{k.split('_')[1]}_days")}
             for k, v in years.items() if k.startswith("year_") and k.endswith("_return")}
    return {"ann_return_net": metrics["ann_return_net"], "total_return_net":
            metrics["total_return_net"], "max_dd": metrics["max_dd"],
            "sharpe_net": metrics["sharpe_net"], "n_periods": metrics["n_periods"],
            "ann_return_net_ex_best_1pct": metrics["ann_return_net_ex_best_1pct"],
            "years": dict(sorted(table.items()))}


def counterfactual(session: Session, trial_id: int, rule: Rule, *, label: str | None = None,
                   run: StoredRun | None = None, reselect_book: bool = False) -> Output:
    """The stored run beside the same run with one declared rule applied, on the same rows
    (for a venue-listing rule, from the other venue's first day). Reports yearly return, total,
    worst fall, Sharpe and the year table of both, and how much exposure the rule removed.

    By default what the rule bans simply becomes cash: the strategy is not asked again. With
    `reselect_book` (a ban rule only) the strategy runs once more with the banned coin-days
    untradeable and chooses among the rest, which answers "what if the book could only trade
    these"; the freed money then goes to other coins and the numbers can differ a lot from the
    plain removal."""
    stored = run or load_run(session, trial_id)
    effect = apply_rule(stored, rule)
    base = stored.backtest()
    mode = "the banned positions became cash"
    if rule.max_leg_contribution is not None:
        changed = capped_leg_result(stored, rule.max_leg_contribution)
        mode = "the clipped money is not counted"
    elif reselect_book and effect.blocked is not None:
        changed = reselect(stored, effect.blocked).backtest()
        mode = "the strategy chose again among the coin-days that remained"
    else:
        changed = stored.backtest(effect.weights)
    start = None
    if effect.first_row:
        start = stored.panel.prices.index[effect.first_row]
    a, b = summarize_result(_slice_result(base, start)), summarize_result(
        _slice_result(changed, start))
    window = f"{(start or base.net_return.index[0]).date()} to {base.net_return.index[-1].date()}"
    lines = [f"trial {trial_id} ({stored.idea_id}); rule: {label or rule.describe()}",
             f"mode: {mode}",
             f"rows compared: {window} ({a['n_periods']} bars); the rule touched "
             f"{effect.removed_bars} instrument-bars with a position"
             + (f"; {effect.unknown_bars} held instrument-bars have no volume figure and were "
                "kept" if effect.unknown_bars else ""), ""]
    lines += _table(["", "original", "with the rule"], [
        ["yearly return", _pct(a["ann_return_net"]), _pct(b["ann_return_net"])],
        ["total return", _pct(a["total_return_net"]), _pct(b["total_return_net"])],
        ["worst fall", _pct(a["max_dd"]), _pct(b["max_dd"])],
        ["Sharpe", f"{a['sharpe_net']:.2f}", f"{b['sharpe_net']:.2f}"],
        ["yearly return without the best 1% of days", _pct(a["ann_return_net_ex_best_1pct"]),
         _pct(b["ann_return_net_ex_best_1pct"])]])
    lines += ["", "by calendar year (compounded net return of the year):"]
    years = sorted(set(a["years"]) | set(b["years"]))
    lines += _table(["year", "original", "with the rule", "days"], [
        [str(y), _pct(a["years"].get(y, {}).get("return")),
         _pct(b["years"].get(y, {}).get("return")),
         str(int(a["years"].get(y, {}).get("days") or 0))] for y in years])
    return Output(lines, {"original": a, "with_rule": b, "removed_bars": effect.removed_bars,
                          "unknown_bars": effect.unknown_bars, "rule": rule.describe(),
                          "mode": mode, "from": str((start or base.net_return.index[0]).date())})


BATTERY = (
    ("volume at least $1M a day", Rule(min_volume_usd=1_000_000)),
    ("volume at least $5M a day", Rule(min_volume_usd=5_000_000)),
    ("no position in the 30 days before a delisting", Rule(delist_days=30)),
    ("one coin earns at most 2% of capital a bar", Rule(max_leg_contribution=0.02)),
    ("one coin earns at most 1% of capital a bar", Rule(max_leg_contribution=0.01)),
)


def battery(session: Session, trial_id: int, *, run: StoredRun | None = None,
            reselect_book: bool = False, listed: pd.DataFrame | None = None) -> Output:
    """The standard robustness battery on one trial: the original and each case of `BATTERY`
    (plus "only coins the other venue lists" when the registry has its listing, or `listed`),
    one row each: yearly return, worst fall, yearly return without the best 1% of days, and the
    last three calendar years. A result that moves by a multiple under a small rule is a
    lottery, not a rule (docs/ANOMALIES.md). Read-only."""
    stored = run or load_run(session, trial_id)
    base = stored.backtest()
    cases: list[tuple[str, Rule]] = list(BATTERY)
    mask = listed if listed is not None else listed_mask(session, stored)
    if mask is not None:
        cases.append(("only coins the other venue lists", Rule(only_listed=mask)))
    rows: list[list[str]] = []
    data: dict[str, Any] = {}

    def row(name: str, result: RunResult, start: pd.Timestamp | None = None) -> None:
        s = summarize_result(_slice_result(result, start))
        years = list(s["years"])[-3:]
        rows.append([name, _pct(s["ann_return_net"]), _pct(s["max_dd"]),
                     _pct(s["ann_return_net_ex_best_1pct"]),
                     "  ".join(f"{y}: {_pct(s['years'][y]['return'], 0)}" for y in years)])
        data[name] = s

    row("original", base)
    for name, rule in cases:
        effect = apply_rule(stored, rule)
        start = stored.panel.prices.index[effect.first_row] if effect.first_row else None
        if rule.max_leg_contribution is not None:
            result = capped_leg_result(stored, rule.max_leg_contribution)
        elif reselect_book and effect.blocked is not None:
            result = reselect(stored, effect.blocked).backtest()
        else:
            result = stored.backtest(effect.weights)
        if start is not None:  # a venue's listing starts later: the original on the same rows
            row("original, same rows (from " + str(start.date()) + ")", base, start)
        row(name, result, start)
    lines = [f"trial {trial_id} ({stored.idea_id}); robustness battery, "
             + ("the strategy chose again where a coin-day is banned"
                if reselect_book else "banned positions became cash"), ""]
    lines += _table(["case", "yearly", "worst fall", "yearly w/o best 1% days", "last years"], rows)
    return Output(lines, {"cases": data})


def listed_mask(session: Session, run: StoredRun) -> pd.DataFrame | None:
    """Which of the run's columns the other venue lists, coin-day by coin-day (the venue's own
    tradeable flag), on the run's index and columns; None when the registry has no Hyperliquid
    snapshot."""
    hl = listing_panel(session, run)
    if hl is None:
        return None
    names = hl_names(run, list(hl.prices.columns))
    out = pd.DataFrame(False, index=run.panel.prices.index, columns=run.panel.prices.columns)
    for column, venue in names.items():
        if venue is None:
            continue
        flag = hl.tradeable[venue].reindex(out.index).fillna(False).astype(bool)
        out[column] = flag
    return out


# --------------------------------------------------------------------------
# compare venues


def compare_venues(session: Session, idea_id: str, *, coins: int = 6) -> Output:
    """An idea's stage 1 and stage 2 over their common days: totals, the daily correlation,
    the months side by side, and each stage's best-earning coins over that window with whether
    the other venue lists them."""
    from qlab import forward_series
    from qlab.anomalies.scan import trial_daily_capital

    found = forward_series.stage_runs(session, idea_id)
    if found is None or not found[0] or any(s.trial_id is None for s in found[1]):
        return Output([f"{idea_id} has no two-stage answer (stage 1 and stage 2 both run) in "
                       "the registry"], {})
    s1, s2 = found[1][0], found[1][1]
    # Both stages on the strategy's own money (docs/CAPITAL_BASE.md), summed when it does not
    # reinvest its profit.
    pair = [trial_daily_capital(session, s.trial_id) for s in (s1, s2)]
    if pair[0] is None or pair[1] is None:
        return Output(["a stage's daily series could not be built"], {})
    (a, reinvested_a), (b, reinvested_b) = pair
    if reinvested_a != reinvested_b:
        return Output(["the two stages do not agree on whether the profit is reinvested: "
                       "their totals are not comparable"], {})
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return Output(["the two stages share no days"], {})
    a, b = a.loc[common], b.loc[common]

    def tot(r: pd.Series) -> float:
        return float(r.sum()) if not reinvested_a else float((1 + r).prod() - 1)

    corr = float(a.corr(b)) if a.std() > 0 and b.std() > 0 else float("nan")
    lines = [f"{idea_id}: stage 1 = {s1.idea_id} (trial {s1.trial_id}, {s1.source}), stage 2 = "
             f"{s2.idea_id} (trial {s2.trial_id}, {s2.source})",
             f"common window: {common.min().date()} to {common.max().date()} ({len(common)} days)",
             f"total return over it: stage 1 {_pct(tot(a))}, stage 2 {_pct(tot(b))}; "
             f"correlation of daily returns {corr:.2f}", "",
             "by month (" + ("compounded" if reinvested_a else "summed, on the strategy's own "
                             "money") + "):"]

    def by_month(r: pd.Series) -> pd.Series:
        grouped = r.groupby([r.index.year, r.index.month])
        return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1 if reinvested_a \
            else grouped.sum()

    months = pd.DataFrame({"s1": by_month(a), "s2": by_month(b)})
    lines += _table(["month", "stage 1", "stage 2", "difference"], [
        [f"{y}-{m:02d}", _pct(r.s1), _pct(r.s2), _pct(r.s1 - r.s2)]
        for (y, m), r in months.iterrows()][-MAX_ROWS:])
    data: dict[str, Any] = {"stage1_total": tot(a), "stage2_total": tot(b),
                            "days": len(common), "correlation": corr}
    for number, stage, other in ((1, s1, s2), (2, s2, s1)):
        try:
            run = load_run(session, stage.trial_id)
        except Exception as exc:  # noqa: BLE001 - say why this stage's coins are missing
            lines += ["", f"stage {number}: coins not available ({exc})"]
            continue
        dec = decompose(run)
        net = daily(dec.net, "sum")
        net = net.loc[net.index.intersection(common)]
        best = net.sum().sort_values(ascending=False).head(coins)
        other_run = None
        try:
            other_run = load_run(session, other.trial_id)
        except Exception:  # noqa: BLE001
            pass
        lines += ["", f"stage {number} ({stage.source}): best-earning coins over the common window "
                  f"(share of capital, arithmetic) and whether stage {3 - number}'s data has them:"]
        rows = []
        for name, value in best.items():
            has = "—"
            if other_run is not None:
                from qlab.pipeline.stage2 import venue_name

                has = "yes" if (name in other_run.panel.prices.columns or venue_name(
                    name, list(other_run.panel.prices.columns))) else "no"
            rows.append([name, _pct(value, 2), has])
        lines += _table(["coin", "earned", "in the other stage's data"], rows)
        data[f"stage{number}_best"] = {n: float(v) for n, v in best.items()}
    return Output(lines, data)


# --------------------------------------------------------------------------
# flags


def stored_flags(session: Session, trial_id: int) -> Output:
    """The detector's stored scan of the trial: each flag, its state, number and line."""
    from qlab.anomalies.store import latest_scan

    row = latest_scan(session, trial_id)
    if row is None:
        return Output([f"trial {trial_id} has no stored scan (qlab anomalies scan {trial_id})"], {})
    lines = [f"trial {trial_id} ({row.idea_id}); lines version {row.config_version}; "
             f"fired: {', '.join(row.fired) or 'none'}"]
    for f in row.flags:
        value = "" if f.get("value") is None else f" value {f['value']:.4g}"
        line = "" if f.get("line") is None else f" line {f['line']:.4g}"
        why = f" ({f['reason']})" if f.get("reason") else ""
        lines.append(f"  {f['name']}: {f['state']}{value}{line}{why}")
        if f.get("detail"):
            lines.append("      " + json.dumps(f["detail"], ensure_ascii=False)[:600])
    return Output(lines, {"fired": row.fired})


__all__ = ["BATTERY", "LOG_ENV", "MAX_ROWS", "Output", "battery",
           "compare_venues", "coin", "contributions",
           "counterfactual", "hl_names", "listed_mask", "listing_panel", "log_call",
           "parse_days", "stored_flags", "summarize_result"]
