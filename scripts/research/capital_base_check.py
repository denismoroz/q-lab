"""The percent on the strategy's own money, measured on the stored runs (docs/CAPITAL_BASE.md).

Read only: it needs a COPY of the registry in QLAB_DB (it reads trials, specs and snapshots,
writes nothing), the data snapshots in data/snapshots and the spec files in specs/. CPU only: no
agent, no network.

    QLAB_DB=/path/to/copy.db uv run python scripts/research/capital_base_check.py key
    QLAB_DB=... uv run python scripts/research/capital_base_check.py agree
    QLAB_DB=... uv run python scripts/research/capital_base_check.py rules --noise 100

* `key`   -- the long runs of docs/research/percent-base-2026-10-28.md, re-measured with the
             production code path (`capital_setup` -> `run_backtest(**kwargs)` ->
             `capital_metrics`): the stand's figures and the capital's side by side, the year
             table and the dollar cross-check (the summed capital return in dollars against the
             change of the book's own balance).
* `agree` -- stage 1 against stage 2 for the twins that have both (`qlab.pipeline.confirm`): where
             the second stage's result falls among the first stage's same-length segments, read
             the stand's way and on the strategy's own money.
* `rules` -- step 6, read only: for every idea whose latest run is on a strategy with a
             non-trivial base (Bv2 and its twins, FRAB, spot, XSMOM), what each decisive return
             rule would decide if it read the capital metric. The noise rank is recomputed from
             `--noise` matched noise books of the dollar-neutral series, on both bases.

Nothing is written to the registry. Run with `if __name__ == "__main__"` (worker processes).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Job:
    label: str
    trial_id: int
    spec_path: str
    overrides: tuple[tuple[str, object], ...] = ()
    noise: int = 0


# the runs the percent-base note measured (docs/research/percent-base-2026-10-28.md, 4.1)
KEY = (
    Job("Bv2 whole Binance, shared pool (94487)", 94487,
        "specs/research/bv2/bv2-research-shared-pool-causal.yaml", (("judged_from", None),)),
    Job("Bv2 twin Binance, 513 d (105587)", 105587,
        "specs/research/bv2/bv2-research-headroom-causal.yaml"),
    Job("Bv2 twin Hyperliquid, 173 d (105788)", 105788,
        "specs/generated/stage2/bv2-research-headroom-causal-hl.yaml"),
    Job("Bv2 prod config Binance (101165)", 101165, "specs/bv2-binance.yaml"),
    Job("Bv2 prod config Hyperliquid (100361)", 100361, "specs/bv2-live.yaml"),
    Job("Bv2 twin partial-hedge Binance (107196)", 107196,
        "specs/research/bv2/bv2-research-partial-hedge-causal.yaml"),
    Job("Bv2 twin short-window Binance (107598)", 107598,
        "specs/research/bv2/bv2-research-short-window-causal.yaml"),
    Job("XSMOM retune Binance (85838)", 85838,
        "specs/research/xsmom/xsmom-research-nobull-retune.yaml"),
    Job("XSMOM twin Binance (108000)", 108000,
        "specs/research/xsmom/xsmom-research-nobull-causal.yaml"),
    Job("XSMOM twin Hyperliquid (108201)", 108201,
        "specs/generated/stage2/xsmom-research-nobull-causal-hl.yaml"),
    Job("Trend twin Binance (93080)", 93080, "specs/trend-binance-cap-gated-causal.yaml"),
    Job("FRAB stand run (100562)", 100562, "specs/frab-live.yaml",
        (("params_fixed_at", date(2026, 10, 10)),)),  # the file's date moved on after the run
    Job("FRAB retuned vs default (104984)", 104984,
        "specs/research/frab/frab-research-live-vs-default-retune.yaml"),
    Job("frab-binance-spot-400 (104180)", 104180,
        "specs/research/frab-binance-spot/frab-binance-spot-400.yaml"),
)

# (stage 1 trial, its spec, stage 2 trial, its spec): twins that have both stages
PAIRS = (
    ("bv2-research-headroom-causal", 106794,
     "specs/research/bv2/bv2-research-headroom-causal.yaml", 106995,
     "specs/generated/stage2/bv2-research-headroom-causal-hl.yaml"),
    ("bv2-research-partial-hedge-causal", 107196,
     "specs/research/bv2/bv2-research-partial-hedge-causal.yaml", 107397,
     "specs/generated/stage2/bv2-research-partial-hedge-causal-hl.yaml"),
    ("bv2-research-short-window-causal", 107598,
     "specs/research/bv2/bv2-research-short-window-causal.yaml", 107799,
     "specs/generated/stage2/bv2-research-short-window-causal-hl.yaml"),
    ("bv2-research-shared-pool-causal", 98552,
     "specs/research/bv2/bv2-research-shared-pool-causal.yaml", 98306,
     "specs/generated/stage2/bv2-research-shared-pool-causal-hl.yaml"),
    ("xsmom-research-nobull-causal", 108000,
     "specs/research/xsmom/xsmom-research-nobull-causal.yaml", 108201,
     "specs/generated/stage2/xsmom-research-nobull-causal-hl.yaml"),
)

NON_TRIVIAL = ("LiveBv2", "LiveFrab", "LiveFrabBinanceSpot", "LiveFrabAllBinance",
               "XsmomCrossSectionalMomentum", "FrabFundingHarvest", "BStrategyV2")


def _db_path() -> str:
    from qlab.registry.db import get_db_path

    return get_db_path()


def recorded(trial_id: int) -> tuple[str, dict]:
    con = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)
    try:
        snapshot_id, metrics = con.execute(
            "select snapshot_id, metrics from trial where id = ?", (trial_id,)).fetchone()
    finally:
        con.close()
    return snapshot_id, json.loads(metrics)


def find_capital(params: object) -> float | None:
    """The `capital_usd` a spec's strategy starts with, wherever it sits in the wrappers."""
    if isinstance(params, dict):
        if isinstance(params.get("capital_usd"), int | float):
            return float(params["capital_usd"])
        for value in params.values():
            found = find_capital(value)
            if found is not None:
                return found
    return None


def chain(spec) -> list[str]:
    names, params = [spec.code_ref.split(":")[-1]], spec.params
    while isinstance(params, dict) and "inner_code_ref" in params:
        names.append(str(params["inner_code_ref"]).split(":")[-1])
        params = params.get("inner_params") or params.get("base_params") or {}
    return names


def book_change(strategy, panel, spec, index) -> float | None:
    """The change of the Bv2 book's own balance over the judged index, in dollars: the book
    itself, or, for a twin, each period's chosen candidate over that period."""
    from qlab.strategies.live.bv2 import LiveBv2
    from qlab.strategies.retune import expand_grid

    params = dict(spec.params)
    if "grid" in params:
        if "LiveBv2" not in chain(spec):
            return None
        candidates = expand_grid(params["base_params"], params["grid"])
        choices = strategy.run(panel, params).choices
        edges = list(choices.index) + [index[-1] + pd.Timedelta(hours=1)]
        total = 0.0
        for i, start in enumerate(choices.index):
            name = choices.iloc[i]
            if not isinstance(name, str):
                continue
            lo, hi = max(start, index[0]), min(edges[i + 1] - pd.Timedelta(hours=1), index[-1])
            if hi <= lo:
                continue
            equity = LiveBv2().run(panel, candidates[name]).book_equity
            total += float(equity.loc[hi] - equity.loc[lo])
        return total
    if spec.code_ref.endswith("LiveBv2"):
        equity = LiveBv2().run(panel, params).book_equity
        return float(equity.loc[index[-1]] - equity.loc[index[0]])
    return None


def measure(job: Job) -> dict:
    """Re-run one stored run with the production code and read it on both bases."""
    from qlab.data.snapshot import load_snapshot
    from qlab.harness.costs import CostModel
    from qlab.harness.metrics import capital_metrics, compute_metrics, year_breakdown
    from qlab.harness.run import run_backtest
    from qlab.harness.strategy import capital_setup, own_costs
    from qlab.pipeline.evaluate import (
        _slice_rows,
        complete_book_window,
        cut_truncated_history,
        judged_rows,
        resolve_strategy,
    )
    from qlab.pipeline.spec import load_spec
    from qlab.registry.db import session_scope
    from qlab.rules import loader

    snapshot_id, rec = recorded(job.trial_id)
    spec = load_spec(ROOT / job.spec_path)
    if job.overrides:
        spec = spec.model_copy(update=dict(job.overrides))
    with session_scope() as session:
        panel = load_snapshot(snapshot_id, session=session)
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    lo, hi = judged_rows(spec, panel, coverage, loader.load_latest()).judged
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps)
    extra = own_costs(strategy, panel, spec.params)
    setup = capital_setup(strategy, panel, spec.params)
    whole = (lo, hi) == (0, len(panel.prices.index) - 1)
    part = panel if whole else _slice_rows(panel, lo, hi)
    part_weights = weights if whole else weights.iloc[lo:hi + 1]
    result = run_backtest(part, part_weights, costs, part.funding, extra_cost=extra,
                          **setup.kwargs)
    stand = compute_metrics(result)
    stand.update(year_breakdown(result.net_return))
    cap = capital_metrics(result) if setup.refused is None else {}
    out = {
        "label": job.label, "trial_id": job.trial_id, "spec": job.spec_path,
        "chain": chain(spec), "recorded_n": rec.get("n_periods"),
        "recorded_ann": rec.get("ann_return_net"), "stand": stand, "cap": cap,
        "refused": setup.refused, "reinvested": setup.reinvested,
        "first": str(result.net_return.index[0]), "last": str(result.net_return.index[-1]),
        "net_return": result.net_return, "capital_return": result.capital_return,
        "base_mean": None if setup.base is None else float(setup.base.reindex(
            result.net_return.index).mean()),
        "base_last": None if setup.base is None else float(setup.base.reindex(
            result.net_return.index).iloc[-1]),
    }
    capital = find_capital(spec.params)
    if capital is not None and setup.refused is None and not setup.reinvested:
        out["capital_usd"] = capital
        out["dollars"] = float(result.capital_return.sum()) * capital
        change = book_change(strategy, panel, spec, result.net_return.index)
        out["book_change"] = change
    if job.noise:
        out["noise"] = noise_ranks(spec, panel, lo, hi, costs, stand, cap, job.noise)
    return out


def noise_ranks(spec, panel, lo, hi, costs, stand, cap, n) -> dict:
    """Where the run's yearly return falls among `n` matched noise books of the dollar-neutral
    series, on the stand's base and on the strategy's own money (both from the SAME books, each
    read on the base and the reinvest choice of the strategy it imitates)."""
    from qlab.calibration.noise import NoiseStrategy
    from qlab.calibration.percentile import noise_percentile
    from qlab.calibration.run import GENERATOR_NAMES, build_noise_spec
    from qlab.harness.metrics import capital_metrics, compute_metrics
    from qlab.harness.run import run_backtest
    from qlab.harness.strategy import capital_setup
    from qlab.pipeline.evaluate import _slice_rows

    whole = (lo, hi) == (0, len(panel.prices.index) - 1)
    part = panel if whole else _slice_rows(panel, lo, hi)
    stand_values, cap_values, failed = [], [], 0
    for i in range(n):
        noise_spec = build_noise_spec(series="dollar_neutral",
                                      generator=GENERATOR_NAMES[i % len(GENERATOR_NAMES)],
                                      seed=i, reference=spec)
        strategy = NoiseStrategy()
        try:
            w = strategy.target_weights(panel, noise_spec.params)
            setup = capital_setup(strategy, panel, noise_spec.params)
            result = run_backtest(part, w if whole else w.iloc[lo:hi + 1], costs, part.funding,
                                  **setup.kwargs)
            stand_values.append(compute_metrics(result)["ann_return_net"])
            cap_values.append(capital_metrics(result)["cap_ann_return"])
        except Exception:  # noqa: BLE001 - a book that cannot match the shape is not a sample
            failed += 1
    out = {"n": len(stand_values), "failed": failed}
    if stand_values:
        out["stand_percentile"] = noise_percentile(stand["ann_return_net"], stand_values)
        if cap:
            out["cap_percentile"] = noise_percentile(cap["cap_ann_return"], cap_values)
    return out


# ---- reports --------------------------------------------------------------------------------


def pct(x, digits=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.{digits}f}%"


def num(x) -> str:
    return "" if x is None else format(x, ".2f")


def stand_years(stand: dict) -> dict[int, tuple[float, int]]:
    return {int(k.split("_")[1]): (v, int(stand[f"year_{k.split('_')[1]}_days"]))
            for k, v in stand.items() if k.startswith("year_") and k.endswith("_return")}


def cap_years(cap: dict) -> dict[int, tuple[float, int]]:
    return {int(k.split("_")[2]): (v, int(cap[f"cap_year_{k.split('_')[2]}_days"]))
            for k, v in cap.items() if k.startswith("cap_year_") and k.endswith("_return")}


def describe(r: dict) -> list[str]:
    s, c = r["stand"], r["cap"]
    lines = [f"\n=== {r['label']} ==="]
    n_ok = r["recorded_n"] is None or int(r["recorded_n"]) == int(s["n_periods"])
    ann_ok = r["recorded_ann"] is None or abs(r["recorded_ann"] - s["ann_return_net"]) < 1e-6
    lines.append(f"chain {'>'.join(r['chain'])}; judged {r['first'][:10]} .. {r['last'][:10]}; "
                 f"n_periods {int(s['n_periods'])} (recorded {r['recorded_n']}"
                 f"{'' if n_ok else ' MISMATCH'}); ann {pct(s['ann_return_net'])} (recorded "
                 f"{pct(r['recorded_ann'])}{'' if ann_ok else ' MISMATCH'}); "
                 f"{'summed' if not r['reinvested'] else 'compounded'}"
                 + (f"; base mean {r['base_mean']:.2f}, last {r['base_last']:.2f}"
                    if r["base_mean"] is not None else ""))
    if r["refused"]:
        lines.append(f"base refused: {r['refused']}")
        return lines
    lines.append(f"{'':<22}{'stand':>10}{'capital':>10}")
    rows = [("per year", "ann_return_net", "cap_ann_return", pct),
            ("total", "total_return_net", "cap_total_return", pct),
            ("worst fall", "max_dd", "cap_max_dd", pct),
            ("Sharpe", "sharpe_net", "cap_sharpe", lambda v: f"{v:.2f}"),
            ("ex best 1% days", "ann_return_net_ex_best_1pct", "cap_ann_return_ex_best_1pct", pct),
            ("ex best year", "ann_return_net_ex_best_year", "cap_ann_return_ex_best_year", pct),
            ("best year share", "best_year_share_of_profit", "cap_best_year_share_of_profit",
             lambda v: pct(v, 0)),
            ("years positive", "years_positive_share", "cap_years_positive_share",
             lambda v: pct(v, 0))]
    for name, a, b, fmt in rows:
        va, vb = s.get(a), c.get(b)
        lines.append(f"{name:<22}{fmt(va) if va is not None else '—':>10}"
                     f"{fmt(vb) if vb is not None else '—':>10}")
    sy, cy = stand_years(s), cap_years(c)
    if sy:
        lines.append("by year (days)        stand   capital")
        for year in sorted(sy):
            lines.append(f"  {year} ({sy[year][1]:3d} d)  {pct(sy[year][0]):>9} "
                         f"{pct(cy[year][0]):>9}")
    if "dollars" in r:
        change = r.get("book_change")
        text = f"dollars: summed capital return x ${r['capital_usd']:g} = {r['dollars']:+.2f}"
        if change is not None:
            text += (f"; the book's own balance change {change:+.2f} "
                     f"(difference {100 * (r['dollars'] / change - 1):+.1f}%)")
        lines.append(text)
    if "noise" in r:
        nz = r["noise"]
        lines.append(f"noise rank over {nz['n']} books ({nz['failed']} failed): stand "
                     f"{nz.get('stand_percentile')}, capital {nz.get('cap_percentile')}")
    return lines


def run_jobs(jobs: list[Job], workers: int) -> list[dict]:
    if workers <= 1:
        return [measure(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(measure, jobs))


def key(workers: int, out: Path | None, only: str | None = None) -> None:
    results = run_jobs([j for j in KEY if only is None or only in j.label], workers)
    for r in results:
        print("\n".join(describe(r)))
    if out:
        out.write_text(json.dumps({r["label"]: {"stand": r["stand"], "cap": r["cap"],
                                                "dollars": r.get("dollars"),
                                                "book_change": r.get("book_change")}
                                   for r in results}, indent=1, default=float))


def agree(workers: int, only: str | None = None) -> None:
    """Stage 2's result among stage 1's same-length segments, both ways."""
    from qlab.data.snapshot import load_snapshot  # noqa: F401 - the loader needs the registry
    from qlab.pipeline import confirm
    from qlab.pipeline.spec import load_spec
    from qlab.registry.db import session_scope

    pairs = [p for p in PAIRS if only is None or only in p[0]]
    jobs = [Job(f"{name} stage 2", t2, spec2) for name, _, _, t2, spec2 in pairs]
    stage2 = {r["trial_id"]: r for r in run_jobs(jobs, workers)}
    print(f"{'idea':<36}{'days':>5}{'stage 2 stand':>15}{'stage 2 cap':>13}"
          f"{'stand: worse':>14}{'cap: worse':>12}{'recorded':>10}")
    for name, t1, spec1, t2, spec2 in pairs:
        r2 = stage2[t2]
        with session_scope() as session:
            history = confirm.load_stage1_history(session, load_spec(ROOT / spec1), t1)
        days = confirm.window_days_of(r2["stand"], load_spec(ROOT / spec2).data.interval)
        from qlab import decay

        stand_total = r2["stand"]["total_return_net"]
        cap_total = r2["cap"]["cap_total_return"]
        seg_stand = confirm.segments_of(history.net_return, days)
        seg_cap = confirm.segments_of(history.capital_return, days,
                                      additive=not history.reinvested)
        p_stand = decay.percentile_of(stand_total, list(seg_stand.totals))
        p_cap = decay.percentile_of(cap_total, list(seg_cap.totals))
        con = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)
        rec = con.execute("select json_extract(metrics,'$.stage1_segment_percentile') "
                          "from trial where id=?", (t2,)).fetchone()[0]
        con.close()
        print(f"{name:<36}{days:>5}{pct(stand_total):>15}{pct(cap_total):>13}"
              f"{p_stand:>14.1f}{p_cap:>12.1f}{'' if rec is None else format(rec, '.1f'):>10}")
        print(f"    stage 1 windows: stand median {pct(float(np.median(seg_stand.totals)))}, "
              f"capital median {pct(float(np.median(seg_cap.totals)))}; "
              f"{len(seg_cap.totals)} windows, history {seg_cap.history_from} .. "
              f"{seg_cap.history_to}")


def rules(workers: int, noise: int, out: Path | None) -> None:
    """Step 6, read only: what the decisive return rules would decide on the capital metric."""
    from qlab.forward_series import spec_file_for
    from qlab.pipeline.evaluate import _spec_row_content  # noqa: F401
    from qlab.registry.db import session_scope

    con = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)
    rows = con.execute("""
        select i.id, i.status, t.id, t.route, s.id, s.code_ref, s.params,
               json_extract(t.metrics,'$.noise_return_percentile')
        from idea i join spec s on s.idea_id = i.id join trial t on t.spec_id = s.id
        where i.id not like 'noise-%' and t.route is not null and t.status = 'ok'
          and t.metrics is not null
          and t.id = (select max(t2.id) from trial t2 join spec s2 on s2.id = t2.spec_id
                      where s2.idea_id = i.id and t2.route is not null and t2.status = 'ok'
                      and t2.metrics is not null)
        order by i.id""").fetchall()
    con.close()
    jobs, meta, skipped = [], {}, []
    for idea_id, status, trial_id, route, spec_row_id, code_ref, params, noise_rec in rows:
        names = [code_ref.split(":")[-1]]
        p = json.loads(params)
        while isinstance(p, dict) and "inner_code_ref" in p:
            names.append(str(p["inner_code_ref"]).split(":")[-1])
            p = p.get("inner_params") or p.get("base_params") or {}
        if not any(n in NON_TRIVIAL for n in names):
            continue
        with session_scope() as session:
            path = spec_file_for(session, spec_row_id, ROOT / "specs")
        if path is None:
            skipped.append((idea_id, trial_id, "no spec file that matches the stored row"))
            continue
        jobs.append(Job(idea_id, trial_id, str(path.relative_to(ROOT)), (), noise))
        meta[idea_id] = (status, route, noise_rec)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = {j.label: pool.submit(measure, j) for j in jobs}
        for label, fut in futures.items():
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - one idea failing must not hide the rest
                skipped.append((label, 0, f"{type(exc).__name__}: {exc}"))
    print(f"{'idea':<44}{'status':<10}{'route':<21}{'stand/yr':>9}{'cap/yr':>9}"
          f"{'edge>0':>8}{'cap>0':>7}{'noise rec':>10}{'noise std':>10}{'noise cap':>10}")
    table = []
    for r in sorted(results, key=lambda r: r["label"]):
        status, route, noise_rec = meta[r["label"]]
        s, c = r["stand"], r["cap"]
        a, b = s["ann_return_net"], c.get("cap_ann_return")
        nz = r.get("noise", {})
        row = {"idea": r["label"], "status": status, "route": route, "trial": r["trial_id"],
               "chain": ">".join(r["chain"]), "stand_ann": a, "cap_ann": b,
               "edge_stand": a > 0, "edge_cap": None if b is None else b > 0,
               "noise_recorded": noise_rec, "noise_stand": nz.get("stand_percentile"),
               "noise_cap": nz.get("cap_percentile"), "n_ok": r["recorded_n"] is None
               or int(r["recorded_n"]) == int(s["n_periods"]),
               "reinvested": r["reinvested"]}
        table.append(row)
        print(f"{row['idea']:<44}{status:<10}{route:<21}{pct(a):>9}{pct(b):>9}"
              f"{str(row['edge_stand']):>8}{str(row['edge_cap']):>7}"
              f"{num(noise_rec):>10}{num(nz.get('stand_percentile')):>10}"
              f"{num(nz.get('cap_percentile')):>10}")
    for idea_id, trial_id, why in skipped:
        print(f"SKIPPED {idea_id} (trial {trial_id}): {why}")
    if out:
        out.write_text(json.dumps({"table": table, "skipped": skipped}, indent=1, default=float))


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["key", "agree", "rules"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--noise", type=int, default=0)
    parser.add_argument("--only", default=None,
                        help="key / agree: a substring of the label / idea name")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.what == "key":
        key(args.workers, args.out, args.only)
    elif args.what == "agree":
        agree(args.workers, args.only)
    else:
        rules(args.workers, args.noise, args.out)


if __name__ == "__main__":
    main()
