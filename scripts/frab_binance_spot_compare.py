"""Side-by-side numbers for docs/FRAB_BINANCE_SPOT.md: FRAB live (spot on
Hyperliquid) against FRAB with the spot on Binance, on the same bars.

Usage: ``uv run python scripts/frab_binance_spot_compare.py SPEC [SPEC...] [--set a.b=v ...]``

Reads each spec's cached panel (no download), cuts it at the same honest start the
stand does (the venue serves only its last 5000 hourly candles), runs the spec's
strategy and the harness, and prints, per spec, the whole window and its two parts
(before and from `params_fixed_at`): annual return net, Sharpe, maximum drawdown,
and the decomposition `net = gross + funding - costs` as sums of per-bar returns,
plus how many coins were enterable and how many positions were open on average.
Nothing is written to the registry: these are measurements of runs the registry
already holds (trials of the same specs), not new trials.

``--set`` overrides a param (``params.concurrency_cap=8``) for a measurement; the
registry-writing sensitivity runs are `scripts/frab_binance_spot_grid.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics
from qlab.harness.run import run_backtest
from qlab.harness.strategy import own_costs
from qlab.pipeline.evaluate import (
    _slice_rows,
    resolve_cached_panel,
    resolve_strategy,
    split_at_fixed_date,
)
from qlab.pipeline.spec import load_spec
from qlab.registry.db import session_scope


def honest_panel(session, spec):
    panel = resolve_cached_panel(session, spec)
    if panel is None:
        raise SystemExit(f"{spec.idea_id}: no cached snapshot answers this spec's data block")
    truncated = dict(panel.meta.get("history_truncated") or {})
    if truncated:
        cut = int(panel.prices.index.searchsorted(pd.Timestamp(max(truncated.values()))))
        panel = _slice_rows(panel, cut, len(panel.prices.index) - 1)
    return panel


def part_numbers(result, lo: int, hi: int) -> dict[str, float]:
    sliced = type(result)(
        snapshot_id=result.snapshot_id,
        net_return=result.net_return.iloc[lo:hi + 1],
        gross_return=result.gross_return.iloc[lo:hi + 1],
        turnover=result.turnover.iloc[lo:hi + 1], cost=result.cost.iloc[lo:hi + 1],
        accrual=result.accrual.iloc[lo:hi + 1])
    m = compute_metrics(sliced)
    days = len(sliced.net_return) / 24
    return {"days": days, "ann_return": m["ann_return_net"], "sharpe": m["sharpe_net"],
            "max_dd": m["max_dd"],
            "total_net": float((1 + sliced.net_return).prod() - 1),
            "gross": float(sliced.gross_return.sum()), "funding": float(sliced.accrual.sum()),
            "costs": float(sliced.cost.sum())}


def measure(spec, overrides: dict[str, object] | None = None) -> dict:
    with session_scope() as session:
        panel = honest_panel(session, spec)
    params = {**spec.params, **(overrides or {})}
    strategy = resolve_strategy(spec.code_ref)
    run = strategy.run(panel, params) if hasattr(strategy, "run") else None
    weights = run.weights if run is not None else strategy.target_weights(panel, params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps)
    extra = own_costs(strategy, panel, params)
    result = run_backtest(panel, weights, costs, panel.funding, extra_cost=extra)
    index = panel.prices.index
    last = len(result.net_return) - 1
    split = split_at_fixed_date(index, 0, len(index) - 1, spec.params_fixed_at,
                                spec.params_fit_from)
    out = {"idea": spec.idea_id, "start": index[0].date(), "end": index[-1].date(),
           "whole": part_numbers(result, 0, last)}
    if split.selection is not None:
        out["before"] = part_numbers(result, split.selection[0], min(split.selection[1], last))
    if split.forward is not None:
        out["forward"] = part_numbers(result, split.forward[0], min(split.forward[1], last))
    if run is not None and run.eligible is not None:
        live = run.eligible[run.eligible > 0]
        out["eligible_mean"] = float(run.eligible.iloc[:last + 1].mean())
        out["eligible_max"] = float(run.eligible.max())
        out["held_mean"] = float(run.held.iloc[:last + 1].mean())
        out["held_share_full"] = float((run.held >= params.get("concurrency_cap", 1)).mean())
        out["first_eligible"] = live.index[0] if len(live) else None
        out["positions"] = len(run.positions)
        out["diagnostics"] = run.diagnostics
    return out


def show(r: dict) -> None:
    print(f"\n== {r['idea']}  {r['start']} .. {r['end']}")
    for part in ("whole", "before", "forward"):
        if part in r:
            p = r[part]
            print(f"  {part:<8} {p['days']:6.1f} d  ann {p['ann_return']:+8.2%}  "
                  f"sharpe {p['sharpe']:7.2f}  maxdd {p['max_dd']:+7.2%}  "
                  f"net {p['total_net']:+7.3%}  =  gross {p['gross']:+7.3%}"
                  f" + funding {p['funding']:+7.3%} - costs {p['costs']:7.3%}")
    if "eligible_mean" in r:
        print(f"  coins enterable on average {r['eligible_mean']:.1f} "
              f"(max {r['eligible_max']:.0f}); open on average {r['held_mean']:.2f}; "
              f"positions opened {r['positions']}")
        if r["diagnostics"]:
            print("  diagnostics:", {k: round(v, 4) for k, v in r["diagnostics"].items()})


def grid() -> None:
    """The sensitivity cells of `frab_binance_spot_grid.py` on the whole window (the
    trials in the registry report the judged forward part, 14 days)."""
    import frab_binance_spot_grid as g

    print(f"{'spec':<28}{'slots':>5}{'slot $':>8}  {'extra':<28}{'ann':>8}{'sharpe':>8}"
          f"{'maxdd':>8}{'funding':>9}{'costs':>8}{'held':>6}{'transfer $':>11}")
    for which, slots, footprint, extra in g.CELLS:
        r = measure(g.cell_spec(which, slots, footprint, extra))
        w = r["whole"]
        diag = r.get("diagnostics") or {}
        print(f"{which[1]:<28}{slots:>5}{footprint:>8g}  {str(extra):<28}"
              f"{w['ann_return']:>+8.1%}{w['sharpe']:>8.2f}{w['max_dd']:>+8.1%}"
              f"{w['funding']:>+9.2%}{w['costs']:>8.2%}{r['held_mean']:>6.2f}"
              f"{diag.get('transfer_fees_usd', 0.0):>11.1f}", flush=True)


def main(argv: list[str]) -> None:
    if argv[:1] == ["--grid"]:
        grid()
        return
    paths = [a for a in argv if not a.startswith("--set")]
    sets = {}
    if "--set" in argv:
        for kv in argv[argv.index("--set") + 1:]:
            key, value = kv.split("=", 1)
            sets[key.removeprefix("params.")] = yaml_value(value)
        paths = argv[:argv.index("--set")]
    for path in paths:
        show(measure(load_spec(Path(path)), sets))


def yaml_value(text: str) -> object:
    import yaml

    return yaml.safe_load(text)


if __name__ == "__main__":
    main(sys.argv[1:])
