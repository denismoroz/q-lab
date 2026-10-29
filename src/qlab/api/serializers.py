"""Row -> JSON conversion for the read-only registry API.

Nothing here computes a verdict, a threshold or a metric: every field is
copied from the stored row. The one derived field is `kind`, and it is a
*classification of the row's own shape*, not a judgement — docs/REGISTRY.md
defines exactly three legitimate verdict rows, and the UI is required to
render them differently. Deriving `kind` once, on the server, from the same
columns the DB CheckConstraints police is the only way to guarantee the
frontend cannot accidentally paint a measurement as a decision.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Any

from qlab.capital_text import basis_ru, differs, headline
from qlab.protection import view_of_trial as protection_view
from qlab.registry.models import Driver, Idea, Spec, Trial, Verdict
from qlab.year_view import year_view

# The three legitimate verdict kinds (docs/REGISTRY.md, "Три законных вида
# строки в `verdict`"). Russian labels live in the frontend; these ids are
# the contract.
VERDICT_DECISION = "decision"
VERDICT_UNKNOWN = "unknown"
VERDICT_MEASUREMENT = "measurement"


def verdict_kind(verdict: Verdict) -> str:
    """Classify a verdict row as decision / unknown / measurement.

    - **decision**: a rule was applied to a computed metric, so `passed` is
      set (the DB guarantees `value`/`comparator`/`threshold` are set too).
    - **unknown**: a rule exists but its metric was never computed — `value`
      and `passed` are both null. This is neither a pass nor a fail.
    - **measurement**: a number with no criterion behind it (`comparator`
      and `threshold` null, `source='imported'`, conventionally
      `rule_id='unidentified'`). Presenting it as a decision would require
      inventing the threshold that was never there.

    The three cases are exhaustive under the table's CheckConstraints; the
    final branch exists so a row that somehow escaped them is labelled
    `unknown` (the kind that asserts the least) rather than silently
    rendered as a decision.
    """
    if verdict.passed is not None:
        return VERDICT_DECISION
    if verdict.value is not None and verdict.comparator is None:
        return VERDICT_MEASUREMENT
    return VERDICT_UNKNOWN


def _iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def driver_json(driver: Driver | None) -> dict[str, Any] | None:
    if driver is None:
        return None
    return {
        "id": driver.id,
        "title": driver.title,
        "description": driver.description,
        "kill_condition": driver.kill_condition,
        "observable": driver.observable,
    }


def idea_row_json(idea: Idea, driver: Driver | None = None) -> dict[str, Any]:
    """Idea as shown in the registry list."""
    return {
        "id": idea.id,
        "title": idea.title,
        "status": idea.status.value,
        "shutdown_cause": idea.shutdown_cause.value if idea.shutdown_cause else None,
        "profile": idea.profile.value,
        "asset_class": idea.asset_class.value,
        "source_type": idea.source_type.value,
        "source_url": idea.source_url,
        "driver_id": idea.driver_id,
        "driver_title": driver.title if driver is not None else None,
        "parent_id": idea.parent_id,
        "created_at": _iso(idea.created_at),
        "updated_at": _iso(idea.updated_at),
    }


def idea_detail_json(idea: Idea, driver: Driver | None) -> dict[str, Any]:
    detail = idea_row_json(idea, driver)
    detail.update(
        {
            "claimed_edge": idea.claimed_edge,
            "notes": idea.notes,
            "driver": driver_json(driver),
        }
    )
    return detail


def spec_json(spec: Spec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "idea_id": spec.idea_id,
        "version": spec.version,
        "params": spec.params,
        "data_requirements": spec.data_requirements,
        "rebalance": spec.rebalance,
        "costs_model": spec.costs_model,
        "code_ref": spec.code_ref,
        "created_at": _iso(spec.created_at),
    }


def verdict_json(verdict: Verdict) -> dict[str, Any]:
    """A verdict row, tagged with its kind.

    `passed` is passed through exactly as stored and is never recomputed —
    docs/REGISTRY.md's aggregate is fail-closed, and a frontend that
    re-derived it from the rows it happens to have loaded would not be.
    """
    return {
        "id": verdict.id,
        "idea_id": verdict.idea_id,
        "spec_id": verdict.spec_id,
        "trial_id": verdict.trial_id,
        "kind": verdict_kind(verdict),
        "stage": verdict.stage.value,
        "rule_id": verdict.rule_id,
        "rules_version": verdict.rules_version,
        "metric": verdict.metric,
        "value": verdict.value,
        "comparator": verdict.comparator,
        "threshold": verdict.threshold,
        "passed": verdict.passed,
        "data_range_start": _iso(verdict.data_range_start),
        "data_range_end": _iso(verdict.data_range_end),
        "decided_at": _iso(verdict.decided_at),
        "note": verdict.note,
        "source": verdict.source.value,
    }


@lru_cache(maxsize=1)
def _latest_ruleset():
    """The newest ruleset, for its informational readings only; None if it cannot be read
    (the console then shows the year view without the readings)."""
    from qlab.rules.loader import load_latest

    try:
        return load_latest()
    except Exception:  # noqa: BLE001 - a view must not fail over a missing rules file
        return None


def _numbers(h: Any) -> dict[str, Any]:
    return {"ann_return": h.ann, "sharpe": h.sharpe, "max_dd": h.max_dd, "total_return": h.total,
            "ann_return_ex_best_1pct": h.ex_best_1pct, "noise_percentile": h.noise}


def headline_json(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """The numbers the console leads with: the run on the strategy's own money
    (docs/CAPITAL_BASE.md, owner 2026-10-29) when it has them, else the stand's, and which one
    this is (`on_capital`).
    `stand` holds the stand's share of the book's balance for the small line beside them, and
    `differs` says whether it says something other than the leading figures. `fit` is the
    selection period's yearly figures on the same base. Copied from the stored metrics."""
    if not metrics:
        return None
    h = headline(metrics)
    fit = headline(metrics, "fit_")
    stand = headline({k: v for k, v in metrics.items() if not k.startswith(("cap_", "noise_cap_"))})
    return {
        **_numbers(h),
        "on_capital": h.on_capital,
        "reinvested": h.reinvested,
        "basis": basis_ru(metrics),
        "base_refused": bool(metrics.get("cap_base_refused")),
        "fit": None if fit.ann is None else {
            "ann_return": fit.ann, "sharpe": fit.sharpe, "max_dd": fit.max_dd},
        "stand": _numbers(stand) if h.on_capital else None,
        "differs": differs(metrics),
    }


def trial_json(
    trial: Trial,
    *,
    idea_id: str | None = None,
    idea_title: str | None = None,
    code_ref: str | None = None,
    spec_version: int | None = None,
) -> dict[str, Any]:
    """A trial row plus the routing that says which idea it belongs to.

    `kept` is carried through because docs/REGISTRY.md is explicit that a
    discarded run still counts as a trial: the ledger must show the runs
    that were thrown away, or the deflation denominator is a lie.
    """
    return {
        "id": trial.id,
        "spec_id": trial.spec_id,
        "spec_version": spec_version,
        "idea_id": idea_id,
        "idea_title": idea_title,
        "code_ref": code_ref,
        "config_hash": trial.config_hash,
        "snapshot_id": trial.snapshot_id,
        "code_sha": trial.code_sha,
        "started_at": _iso(trial.started_at),
        "finished_at": _iso(trial.finished_at),
        "metrics": trial.metrics,
        # The numbers to lead with, on the strategy's own money (docs/CAPITAL_BASE.md), the stand's
        # beside them; copied from the stored metrics.
        "headline": headline_json(trial.metrics),
        # The metrics' per-year keys regrouped for the console (docs/YEARS.md); null for a
        # run from before the view existed. Copied from the stored metrics, nothing computed.
        "year_view": year_view(trial.metrics, _latest_ruleset()),
        # The protection readings of a Bv2-family run (docs/PROTECTION.md): P1 - P4 against the
        # lines of the ruleset, recorded and deciding nothing; null for any other run.
        "protection": protection_view(trial.id, trial.metrics, _latest_ruleset()),
        "status": trial.status.value,
        # Null for runs made before routes were stored (docs/TASKS.md T31).
        "route": trial.route.value if trial.route is not None else None,
        "route_reason": trial.route_reason,
        "kept": trial.kept,
        "token_cost": trial.token_cost,
        "cpu_seconds": trial.cpu_seconds,
        "source": trial.source.value,
    }


__all__ = [
    "VERDICT_DECISION",
    "VERDICT_MEASUREMENT",
    "VERDICT_UNKNOWN",
    "driver_json",
    "headline_json",
    "idea_detail_json",
    "idea_row_json",
    "spec_json",
    "trial_json",
    "verdict_json",
    "verdict_kind",
]
