"""Did an overlay help? A variant against its base, on the same days (docs/OVERLAYS.md).

Owner, 2026-10-28: «ну ты вот взял проверил один вариант с потолка и зарубил его» -- no single
variant decides anything, and the answer must be a number the stand computes, not an opinion.
The stand's own run of a variant gives the usual verdict (rules, noise, deflation); this module
adds the question the owner actually asks, **"is the variant better than the same strategy
without the overlay?"**, and answers it by code.

**The days.** Both specs run on the same data (a variant is built on its base's `data`), and each
has its own judged part (`qlab.pipeline.evaluate.judged_rows`: the forward test, or for a causal
variant from its first choice). The comparison is made on the days BOTH are judged on: from the
later of the two starts to the end. Each side is back-tested on exactly that slice, the way the
stand tests a judged part (`run_backtest` on the slice), so the two are like with like.

**The money.** Both sides are read on the strategy's own money (docs/CAPITAL_BASE.md, owner,
2026-10-29: «было бы здорово пофиксить это везде»): each day's return is a share of the capital
the strategy started with, summed over days for a strategy that does not reinvest (Bv2), compounded
for one that does. A variant takes its base and its choice from the strategy it wraps, so the two
sides share one base. A strategy whose base is undefined (Bv2 with deposits), or a pair that
disagrees about reinvesting, is compared on the stand's shares as before and the record says so
(`on_capital` false).

**The test.** The daily net returns of the two are paired by day. The statistic is the
difference of their annualised Sharpe ratios (variant minus base), the stand's own risk-adjusted
measure. Its uncertainty comes from a **paired moving-block bootstrap**: days are resampled in
blocks (circular, of `n ** (1/3)` days, the usual rule of thumb) the same way for both series, so
the dependence between neighbouring days and between the two series is kept; 2000 resamples give
a 95% interval of the difference and a two-sided p-value (twice the smaller tail share on
either side of zero). If the interval includes zero the difference is **indistinguishable from
chance**. The same resampling gives the interval of the difference in the annual return and of the
mean daily difference; they are shown, the verdict reads the Sharpe.

**Many variants, one chance.** A base has up to fifteen variants (every single overlay and every
pair); at the 5% level one in twenty tests passes by luck. The verdict therefore multiplies each
p-value by the number of variants of the base tested so far (Bonferroni, the simplest and the
most cautious correction) before comparing it with 5%. The raw p-value is shown beside it. The
number of trials the base's family stands on (`family_trials`) is shown too: deflation counts
every variant, every run (docs/T27_DEFLATION.md).

Three answers: «помогла» (better, and the corrected p-value is below 5%), «не помогла» (worse
in the same sense), «неотличимо от случайности» (anything else, including a window shorter than a
year). Code only; a verdict here changes no status and no rule reads it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.pipeline.spec import StrategySpec

STORE = Path("data/overlays")
N_BOOT = 2000
SEED = 20261028
DAYS_PER_YEAR = 365.25
ALPHA = 0.05
"""The usual 5% level, applied after the correction for the number of variants."""
MIN_DAYS = 365
"""A shorter common window is too short to tell a difference from chance: the rules judge by
calendar years (docs/YEARS.md)."""
HELPED, NOT_HELPED, UNDISTINGUISHABLE = "helped", "not-helped", "indistinguishable"
VERDICT_RU = {HELPED: "помогла", NOT_HELPED: "не помогла",
              UNDISTINGUISHABLE: "неотличимо от случайности"}


def store_dir() -> Path:
    import os

    return Path(os.environ.get("QLAB_OVERLAYS") or STORE)


# --------------------------------------------------------------------------
# The daily series of a judged part


@dataclass(frozen=True)
class Prepared:
    """A spec run on its snapshot: everything needed to back-test any slice of it."""

    spec: StrategySpec
    panel: object
    weights: pd.DataFrame
    costs: object
    extra_cost: pd.Series | None
    judged_from_row: int
    """The first row of the part the spec is judged on (`judged_rows`)."""
    capital: object = None
    """The strategy's `CapitalSetup` (its own money's base and declared reinvest choice)."""


def prepare(spec: StrategySpec, session, ruleset=None) -> Prepared:
    """Run `spec`'s strategy on the snapshot already on disk (never fetches) and find the first
    row it is judged on."""
    from qlab.harness.costs import CostModel
    from qlab.harness.strategy import capital_setup, own_costs
    from qlab.pipeline.evaluate import (
        complete_book_window,
        cut_truncated_history,
        judged_rows,
        resolve_cached_panel,
        resolve_strategy,
    )
    from qlab.rules.loader import load_latest

    panel = resolve_cached_panel(session, spec)
    if panel is None:
        raise ValueError(f"{spec.idea_id}: no snapshot on disk answers its data request "
                         f"({spec.data.source} {spec.data.start}..{spec.data.end})")
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    rows = judged_rows(spec, panel, coverage, ruleset or load_latest())
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    return Prepared(spec=spec, panel=panel, weights=weights, costs=costs,
                    extra_cost=own_costs(strategy, panel, spec.params),
                    judged_from_row=rows.judged[0],
                    capital=capital_setup(strategy, panel, spec.params))


def daily_from(run: Prepared, first_row: int, *, on_capital: bool = False
               ) -> tuple[pd.Series, pd.Series | None]:
    """The strategy's net return per UTC day from `first_row` to the end of the panel, and the
    same days' return of simply holding the coins it holds (None when it holds nothing).

    With `on_capital` the first series is the strategy's own money's (docs/CAPITAL_BASE.md): each
    day's share of the capital it started with, the days of a strategy that does not reinvest
    summed within a day instead of compounded. The held-coins series is the same either way
    (a basket fully in the coins, a description of its own)."""
    from qlab.harness.metrics import hold_returns
    from qlab.harness.run import run_backtest
    from qlab.pipeline.evaluate import _slice_rows

    last = len(run.panel.prices.index) - 1
    part_panel = _slice_rows(run.panel, first_row, last)
    part_weights = run.weights.iloc[first_row:last + 1]
    kwargs = run.capital.kwargs if on_capital else {}
    result = run_backtest(part_panel, part_weights, run.costs, part_panel.funding,
                          extra_cost=run.extra_cost, **kwargs)
    net = _per_day(result.capital_return if on_capital else result.net_return,
                   additive=on_capital and not result.capital_reinvested)
    held = hold_returns(part_panel.prices, part_weights, result.net_return.index)
    return net, None if held is None else _per_day(held).reindex(net.index).fillna(0.0)


def _per_day(returns: pd.Series, additive: bool = False) -> pd.Series:
    days = pd.DatetimeIndex(returns.index).tz_convert("UTC").normalize()
    if additive:
        return returns.astype(float).groupby(days).sum()
    return (1.0 + returns.astype(float)).groupby(days).prod() - 1.0


# --------------------------------------------------------------------------
# Numbers of one daily series


def sharpe(daily: np.ndarray) -> float:
    std = daily.std(ddof=1)
    return float(daily.mean() / std * math.sqrt(DAYS_PER_YEAR)) if std > 1e-15 else 0.0


def window_metrics(daily: pd.Series, additive: bool = False) -> dict[str, float]:
    """Total, annual return, worst fall from a peak and Sharpe of a daily return series.
    `additive`: the days are shares of a constant capital and add up (a strategy that does not
    reinvest); the worst fall is then in points of that capital."""
    values = daily.to_numpy(dtype=float)
    if additive:
        total = float(values.sum())
        curve = np.cumsum(values)
        return {
            "days": len(values),
            "total_return": total,
            "ann_return": total * DAYS_PER_YEAR / len(values),
            "max_dd": float((curve - np.maximum.accumulate(np.maximum(curve, 0.0))).min()),
            "sharpe": sharpe(values),
        }
    growth = np.cumprod(1.0 + values)
    total = float(growth[-1] - 1.0)
    days = len(values)
    peak = np.maximum.accumulate(np.maximum(growth, 1.0))
    return {
        "days": days,
        "total_return": total,
        "ann_return": -1.0 if growth[-1] <= 0 else float(growth[-1] ** (DAYS_PER_YEAR / days) - 1),
        "max_dd": float((growth / peak - 1.0).min()),
        "sharpe": sharpe(values),
    }


def _block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving blocks: `n_boot` resamples of `n` day positions."""
    count = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, count))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n
            ).reshape(n_boot, count * block)[:, :n]


def _sharpe_rows(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=1, ddof=1)
    mean = values.mean(axis=1)
    return np.where(std > 1e-15, mean / np.where(std > 1e-15, std, 1.0), 0.0
                    ) * math.sqrt(DAYS_PER_YEAR)


def _annual_rows(values: np.ndarray, additive: bool = False) -> np.ndarray:
    if additive:
        return values.sum(axis=1) * DAYS_PER_YEAR / values.shape[1]
    growth = np.prod(1.0 + values, axis=1)
    return np.where(growth > 0, np.abs(growth) ** (DAYS_PER_YEAR / values.shape[1]) - 1.0, -1.0)


def _two_sided(draws: np.ndarray) -> float:
    """Twice the smaller share of resampled differences on either side of zero."""
    below = (np.sum(draws <= 0) + 1) / (len(draws) + 1)
    above = (np.sum(draws >= 0) + 1) / (len(draws) + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def bootstrap_difference(base: pd.Series, variant: pd.Series, *, n_boot: int = N_BOOT,
                         seed: int = SEED, additive: bool = False) -> dict[str, object]:
    """The paired moving-block bootstrap of variant minus base (module docstring)."""
    if not base.index.equals(variant.index):
        raise ValueError("the two series must be on the same days")
    a = base.to_numpy(dtype=float)
    b = variant.to_numpy(dtype=float)
    n = len(a)
    block = max(2, round(n ** (1 / 3)))
    idx = _block_indices(n, block, n_boot, np.random.default_rng(seed))
    sharpe_d = _sharpe_rows(b[idx]) - _sharpe_rows(a[idx])
    annual_d = _annual_rows(b[idx], additive) - _annual_rows(a[idx], additive)
    mean_d = (b[idx] - a[idx]).mean(axis=1) * DAYS_PER_YEAR

    def interval(draws: np.ndarray) -> list[float]:
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    return {
        "n_boot": n_boot, "block_days": block, "seed": seed,
        "delta_sharpe_ci": interval(sharpe_d), "delta_sharpe_p": _two_sided(sharpe_d),
        "delta_ann_return_ci": interval(annual_d), "delta_ann_return_p": _two_sided(annual_d),
        "mean_daily_diff_ann_ci": interval(mean_d), "mean_daily_diff_p": _two_sided(mean_d),
    }


def compare_series(base: pd.Series, variant: pd.Series, *, n_boot: int = N_BOOT,
                   seed: int = SEED, additive: bool = False) -> dict[str, object]:
    """Both series on their common days: each one's numbers, the differences and the bootstrap.
    `additive`: both are shares of a constant capital whose days add up."""
    days = base.index.intersection(variant.index)
    base, variant = base.reindex(days), variant.reindex(days)
    if len(days) < 30:
        raise ValueError(f"only {len(days)} common days: nothing to compare")
    mb, mv = window_metrics(base, additive), window_metrics(variant, additive)
    out: dict[str, object] = {
        "start": days[0].date().isoformat(), "end": days[-1].date().isoformat(),
        "days": len(days), "base": mb, "variant": mv,
        "delta_sharpe": mv["sharpe"] - mb["sharpe"],
        "delta_ann_return": mv["ann_return"] - mb["ann_return"],
        "delta_max_dd": mv["max_dd"] - mb["max_dd"],
    }
    out.update(bootstrap_difference(base, variant, n_boot=n_boot, seed=seed, additive=additive))
    return out


def verdict(record: dict, family_size: int, *, alpha: float = ALPHA,
            min_days: int = MIN_DAYS) -> str:
    """«помогла» / «не помогла» / «неотличимо от случайности» (module docstring), by code from
    the stored comparison and the number of variants of the base tested."""
    if int(record.get("days") or 0) < min_days:
        return UNDISTINGUISHABLE
    corrected = min(1.0, float(record["delta_sharpe_p"]) * max(1, family_size))
    if corrected >= alpha:
        return UNDISTINGUISHABLE
    return HELPED if float(record["delta_sharpe"]) > 0 else NOT_HELPED


def corrected_p(record: dict, family_size: int) -> float:
    return min(1.0, float(record["delta_sharpe_p"]) * max(1, family_size))


# --------------------------------------------------------------------------
# One variant against its base


def compare_specs(base: StrategySpec, variant: StrategySpec, session, *, n_boot: int = N_BOOT
                  ) -> dict[str, object]:
    """Run both specs on their snapshots, find the days both are judged on and compare them."""
    run_base = prepare(base, session)
    run_variant = prepare(variant, session)
    if not run_base.panel.prices.index.equals(run_variant.panel.prices.index):
        raise ValueError("the base and the variant are not on the same bars")
    first = max(run_base.judged_from_row, run_variant.judged_from_row)
    # On the strategy's own money when both sides have a base and agree on whether the profit is
    # reinvested; otherwise on the stand's shares, and the record says so.
    on_capital = (run_base.capital.refused is None and run_variant.capital.refused is None
                  and run_base.capital.reinvested == run_variant.capital.reinvested)
    additive = on_capital and not run_base.capital.reinvested
    net_base, _ = daily_from(run_base, first, on_capital=on_capital)
    net_variant, _ = daily_from(run_variant, first, on_capital=on_capital)
    record = compare_series(net_base, net_variant, n_boot=n_boot, additive=additive)
    record.update(on_capital=on_capital, additive=additive, base_id=base.idea_id,
                  variant_id=variant.idea_id,
                  base_judged_from=run_base.panel.prices.index[run_base.judged_from_row]
                  .date().isoformat(),
                  variant_judged_from=run_variant.panel.prices.index[run_variant.judged_from_row]
                  .date().isoformat(),
                  data_end=variant.data.end.isoformat(),
                  computed_at=datetime.now(UTC).isoformat(timespec="seconds"))
    return record


# --------------------------------------------------------------------------
# Files, the family


def record_path(variant_id: str, store: Path | None = None) -> Path:
    return (store or store_dir()) / "compare" / f"{variant_id}.json"


def save_record(record: dict, store: Path | None = None) -> Path:
    path = record_path(str(record["variant_id"]), store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_record(variant_id: str, store: Path | None = None) -> dict | None:
    try:
        return json.loads(record_path(variant_id, store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def variants_of(session, base_id: str) -> list[str]:
    """The overlay variants registered under `base_id`, by id."""
    from qlab.overlay_variants import INFIX
    from qlab.registry.models import Idea

    return sorted(i.id for i in session.query(Idea).filter(Idea.parent_id == base_id)
                  if i.id.startswith(f"{base_id}{INFIX}"))


def family_size(session, base_id: str, store: Path | None = None) -> int:
    """How many variants of the base have been tested (have a stored comparison): the number
    each p-value is corrected by."""
    return sum(load_record(v, store) is not None for v in variants_of(session, base_id))


def family_trials(session, base_id: str) -> int:
    """The trials the base's family stands on: the base's own and every variant's. Deflation
    reads the same count (`qlab.registry.deflation.family_trials`, scope `driver`, where the
    base has a driver: the variants inherit it)."""
    from sqlalchemy import select

    from qlab.registry.models import Spec, Trial

    ids = [base_id, *variants_of(session, base_id)]
    return len(session.execute(select(Trial.id).join(Spec, Spec.id == Trial.spec_id)
                               .where(Spec.idea_id.in_(ids))).all())


def table(session, base_id: str, store: Path | None = None) -> dict:
    """What the console and the report show for a base: its variants, each with its stored
    comparison and the verdict at the current family size."""
    from qlab.registry.models import Idea
    from qlab.registry.stages import latest_routed

    ids = variants_of(session, base_id)
    size = family_size(session, base_id, store)
    rows = []
    for vid in ids:
        record = load_record(vid, store)
        idea = session.get(Idea, vid)
        routed = latest_routed(session, vid)
        row = {"idea_id": vid, "status": idea.status.value,
               "route": None if routed is None else str(routed[0].route.value),
               "trial_id": None if routed is None else routed[0].id, "comparison": None}
        if record is not None:
            row["comparison"] = {
                **{k: record[k] for k in ("start", "end", "days", "base", "variant",
                                          "delta_sharpe", "delta_ann_return", "delta_max_dd",
                                          "delta_sharpe_ci", "delta_sharpe_p")},
                # A record made before 2026-10-29 is on the stand's shares.
                "on_capital": bool(record.get("on_capital", False)),
                "additive": bool(record.get("additive", False)),
                "corrected_p": corrected_p(record, size),
                "verdict": verdict(record, size),
                "verdict_ru": VERDICT_RU[verdict(record, size)],
            }
        rows.append(row)
    return {"base_id": base_id, "family_size": size, "variants": rows,
            "family_trials": family_trials(session, base_id),
            "counts": {v: sum(1 for r in rows if r["comparison"] and
                              r["comparison"]["verdict"] == v)
                       for v in (HELPED, NOT_HELPED, UNDISTINGUISHABLE)}}


def view(session, idea_id: str, store: Path | None = None) -> dict | None:
    """What the console shows on an idea's page: for a base, its variants against it; for a
    variant, its own comparison and the base's table; the base's diagnostics where stored. None
    for an unknown idea. Everything is read from the registry and the stored files."""
    from qlab import overlay_diagnostics
    from qlab.funnel import ROUTE_TITLE
    from qlab.overlay_variants import base_of
    from qlab.registry.models import Idea

    if session.get(Idea, idea_id) is None:
        return None
    base_id = base_of(idea_id) or idea_id
    role = "variant" if base_of(idea_id) else "base"
    shown = table(session, base_id, store)
    for row in shown["variants"]:
        row["route_ru"] = ROUTE_TITLE.get(row["route"] or "", row["route"])
        row["name"] = row["idea_id"][len(base_id):].lstrip("-").removeprefix("ov-")
    diagnostics = overlay_diagnostics.load(base_id, store)
    return {
        "idea_id": idea_id, "role": role, "base_id": base_id, "table": shown,
        "diagnostics": diagnostics,
        "diagnostics_lines": overlay_diagnostics.lines_ru(diagnostics) if diagnostics else None,
        "alpha": ALPHA, "min_days": MIN_DAYS,
    }


__all__ = ["ALPHA", "HELPED", "MIN_DAYS", "NOT_HELPED", "N_BOOT", "UNDISTINGUISHABLE",
           "VERDICT_RU", "bootstrap_difference", "compare_series", "compare_specs", "corrected_p",
           "daily_from", "family_size", "family_trials", "load_record", "prepare",
           "save_record", "sharpe", "store_dir", "table", "variants_of", "verdict",
           "view", "window_metrics"]
