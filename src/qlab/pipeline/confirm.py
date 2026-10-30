"""Stage 2 of the idea generator's evaluation confirms stage 1's verdict where its own
noise comparison cannot decide (rules 2026-10-27.1, docs/TWO_STAGE.md).

Owner, 2026-10-27, «делай все»: the proof comes from stage 1 (Binance perpetuals since
2019-09, years of data, so the comparison with random books has power there); stage 2
(Hyperliquid, data from 2025-01-01, the same mechanism and the same numbers) is a
CONSISTENCY check, not a second independent proof.

On the Hyperliquid window the comparison with random books often has almost no power:
a strategy of the run's own shape truly earning 7% a year passes the 99% bar 6% of the
time (`noise_power`, rules 2026-10-24.2), so the rule `shape_aware_edge` fails good and
empty strategies alike and the idea waits for years. This module is the route reviewer
(`qlab.pipeline.evaluate.evaluate_spec(review_route=...)`) that, only in that case,
replaces the failed noise comparison by two checks on the stage 2 window:

* its total return after costs is above zero;
* it does not fall in the bottom `alert_percentile` of 100 same-length segments of
  stage 1's FORWARD history (the same strategy on Binance). The percentile is the
  owner's alarm for paper books (`night/decay.yaml`, 7) and is read from there; the
  segments and the percentile are `qlab.decay`'s (`rolling_windows`, `percentile_of`,
  `alert_label`), not a second copy.

Everything else judges stage 2 as before: the other rules, the length of the forward
test and regime coverage (`decide_fit_forward_route` is asked about the ruleset WITHOUT
the noise comparison). Where the noise bar has power, it decides as before. Stage 1
must itself have decided (`paper`, `shelf`, `needs-infrastructure`) on a powered noise
comparison; a stage 1 that is "too early" confirms nothing (and `combine` never lets it
pass). Without stage 1's history the old behaviour stands, and the run's reason says why.

Stage 1's history is read back by re-running its backtest on the data snapshot of its
own trial (CPU only: no agent, no tokens, no download, no trial written).

**Both sides on the strategy's own money** (docs/CAPITAL_BASE.md). The stage 2 result and
stage 1's segments are compared as shares of the capital the strategy started with
(`cap_total_return`, and stage 1's `capital_return`), summed rather than compounded for a
strategy that does not reinvest (Bv2). A share of a book whose balance has grown with set-aside
cash is a smaller share of the money the strategy was given, and stage 1's years of history
have grown that balance while stage 2's fresh book has not: the old comparison set stage 2's
result against segments about four times too small (Bv2: «74 из 100» became 42). The
positivity rule reads the total its ruleset names (`Stage2Confirmation.total_metric`): the stand's
`total_return_net` up to rules 2026-10-29.1, `cap_total_return` from 2026-10-29.2 (owner,
2026-10-29, «Да, и правила»), so the rule and the segments read one figure. Under the older rules
a run without `cap_total_return`, or a history without a capital series, is compared the old
way; under the newer ones a run without `cap_total_return` (a base of percent that is undefined)
confirms nothing, and the reason says so.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from qlab import decay
from qlab.data.sources.base import INTERVAL_TO_TIMEDELTA
from qlab.diagnose import equity_from_returns
from qlab.pipeline.evaluate import (
    RouteReview,
    RoutingContext,
    RoutingDecision,
    decide_fit_forward_route,
)
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage

NOISE_METRIC = "noise_return_percentile"
"""The metric of the noise comparison (`shape_aware_edge`) up to rules 2026-10-29.1; from
2026-10-29.2 the ruleset names it (`NoisePower.percentile_metric`)."""
TOTAL_METRIC = "total_return_net"
"""The stage 2 total the positivity rule reads up to rules 2026-10-29.1; from 2026-10-29.2 the
ruleset names it (`Stage2Confirmation.total_metric`)."""
CONSISTENCY_METRIC = "stage1_segment_percentile"
POSITIVE_RULE = "stage2_total_return_positive"
CONSISTENCY_RULE = "stage1_consistency"
STAGE1_CONFIRMS = ("paper", "shelf", "needs-infrastructure")
"""Stage 1 routes that decided on their forward data. `needs-forward` is absent on
purpose: a stage 1 that is still "too early" confirms nothing."""
PASSING = ("paper", "shelf", "needs-infrastructure")
NOTE = "confirmation by stage 1's history (docs/TWO_STAGE.md)"


class Stage1HistoryMissing(Exception):
    """Stage 1's forward history cannot be read back; the message says why."""


@dataclass(frozen=True)
class Stage1History:
    trial_id: int
    net_return: pd.Series
    """Stage 1's net returns from the date its parameters were fixed (forward)."""
    noise_power: float | None
    """The power of stage 1's own noise comparison, from its trial's metrics."""
    capital_return: pd.Series | None = None
    """The same bars as shares of the strategy's own money (None: not known, read the old way)."""
    reinvested: bool = True
    """The strategy's declared choice: True compounds the segments, False sums them."""
    coins_shared: int | None = None
    """How many of stage 1's columns the second venue also lists, when the history was taken on
    those alone (rules from 2026-10-30.1, `RuleSet.venue_split.shared_coins_history`); None: the
    whole history."""
    coins_total: int | None = None
    whole_history_reason: str | None = None
    """Why the history is the whole of stage 1's, when the shared coins were asked for and it
    could not be taken on them."""


def _run_on_shared_coins(spec, panel, venue_names):
    """Stage 1's strategy on the columns of its panel the second venue also lists, as (the run,
    columns kept, columns in all, why). The run is None, with `why`, when that cannot be computed:
    the whole history then stands."""
    from qlab.pipeline.stage2 import venue_name

    names = venue_names() if callable(venue_names) else venue_names
    columns = [str(c) for c in panel.prices.columns]
    keep = [c for c in columns if venue_name(c, names) is not None]
    if len(keep) == len(columns):
        return decay.run_on_panel(spec, panel), len(keep), len(columns), None
    if not keep:
        return None, 0, len(columns), "the second venue lists none of stage 1's coins"
    try:
        run = decay.run_on_panel(spec, panel.restrict(keep))
    except Exception as exc:  # noqa: BLE001 - strategy code is arbitrary; the whole history stands
        return None, len(keep), len(columns), (
            f"the strategy could not be run on the {len(keep)} shared coins: "
            f"{type(exc).__name__}: {exc}")
    if not (run.net_return != 0.0).any():
        return None, len(keep), len(columns), (
            f"the strategy held nothing on the {len(keep)} shared coins")
    return run, len(keep), len(columns), None


def load_stage1_history(session, spec, trial_id: int | None, *,
                        venue_names=None) -> Stage1History:
    """Stage 1's forward net returns: its strategy run again on the data snapshot its
    trial was decided on. Writes nothing and downloads nothing. The noise bar's power is the one
    on the strategy's own money (`noise_cap_power`, rules from 2026-10-29.2) when the trial
    has it, else the stand's `noise_power`.

    `venue_names` (a collection of the second venue's names, or a function returning it) asks for
    the history on the coins the second venue also lists (rules from 2026-10-30.1,
    docs/VENUE_SPLIT.md): Hyperliquid's window is then set against the same market it was run on,
    not against Binance coins Hyperliquid never listed. When that cannot be computed (the venue
    lists none of the coins, the strategy fails or holds nothing on them) the whole history
    stands, and `whole_history_reason` says why."""
    from qlab.data.snapshot import load_snapshot
    from qlab.registry.models import Trial

    if trial_id is None:
        raise Stage1HistoryMissing("no stage 1 trial is recorded")
    trial = session.get(Trial, trial_id)
    if trial is None:
        raise Stage1HistoryMissing(f"stage 1 trial {trial_id} is not in the registry")
    if trial.snapshot_id is None:
        raise Stage1HistoryMissing(f"stage 1 trial {trial_id} has no data snapshot")
    if spec.params_fixed_at is None:
        raise Stage1HistoryMissing(
            "stage 1 has no forward history: the date its parameters were fixed is unknown")
    try:
        panel = load_snapshot(trial.snapshot_id, session=session)
    except (OSError, ValueError) as exc:
        raise Stage1HistoryMissing(
            f"stage 1's data snapshot {trial.snapshot_id} cannot be read: {exc}") from exc
    shared = total = reason = None
    run = None
    try:
        if venue_names is not None:
            run, shared, total, reason = _run_on_shared_coins(spec, panel, venue_names)
            if run is None:
                shared = None
        if run is None:
            run = decay.run_on_panel(spec, panel)
    except decay.StandHistoryMissing as exc:
        raise Stage1HistoryMissing(str(exc)) from exc
    since = pd.Timestamp(spec.params_fixed_at, tz="UTC")
    forward = run.net_return[run.net_return.index >= since]
    capital = run.capital_return[run.capital_return.index >= since]
    stored = trial.metrics or {}
    power = stored.get("noise_cap_power", stored.get("noise_power"))
    return Stage1History(trial_id, forward, None if power is None else float(power),
                         capital_return=capital, reinvested=run.reinvested,
                         coins_shared=shared, coins_total=total, whole_history_reason=reason)


# --------------------------------------------------------------------------
# Same-length segments of stage 1's history


@dataclass(frozen=True)
class Segments:
    totals: tuple[float, ...]
    """Total return of every window of `window_days` days over the history."""
    window_days: int
    history_days: int
    independent: int
    """How many of them fit without overlapping: the windows overlap, so their number
    overstates how much independent evidence there is."""
    history_from: date | None
    history_to: date | None


def segments_of(net_return: pd.Series, window_days: int, additive: bool = False) -> Segments:
    """Every `window_days`-day window of a net-return series (`qlab.decay.rolling_windows`,
    the one the paper check uses). `additive`: the series is the strategy's capital return and
    its profit is not reinvested, so a window's total is the sum of its bars (the gain in
    points of the capital), not a ratio to what the history held when the window began."""
    closes = decay.daily_closes(equity_from_returns(decay.trim_flat_start(net_return), additive))
    windows = decay.rolling_windows(closes, window_days, additive) if len(closes) > 1 else []
    history_days = max(len(closes) - 1, 0)
    return Segments(
        totals=tuple(w.total for w in windows), window_days=window_days,
        history_days=history_days,
        independent=history_days // window_days if window_days else 0,
        history_from=closes.index[0].date() if len(closes) > 1 else None,
        history_to=closes.index[-1].date() if len(closes) > 1 else None)


def window_days_of(metrics: Mapping[str, float], interval: str) -> int | None:
    """The stage 2 window in days: the periods its return compounds over times the bar."""
    n = metrics.get("n_periods")
    bar = INTERVAL_TO_TIMEDELTA.get(interval)
    if n is None or bar is None or n != n:
        return None
    return max(int(round(float(n) * bar / pd.Timedelta(days=1))), 1)


# --------------------------------------------------------------------------
# The decision


@dataclass
class Confirmation:
    """What the check found, stored in the stage 2 result (and, as numbers, on its trial)."""

    status: str
    """confirmed | inconsistent | not-positive | not-applied"""
    reason: str
    rules_version: str
    route: str | None = None
    """The stage 2 route the check gave; None when it was not applied."""
    alert_percentile: float | None = None
    percentile: float | None = None
    """How many of 100 same-length stage 1 segments were worse than stage 2's result."""
    segments: int | None = None
    independent_segments: int | None = None
    window_days: int | None = None
    total_return: float | None = None
    """The stage 2 total the segments were compared with (on the strategy's own money when
    `on_capital`)."""
    on_capital: bool = False
    stage1_trial_id: int | None = None
    stage1_from: str | None = None
    stage1_to: str | None = None
    noise_power: float | None = None
    """Stage 2's noise bar power on its window: why the comparison was replaced."""
    stage1_noise_power: float | None = None
    noise_min_power: float | None = None
    """The power from which the noise comparison decides (rules `noise_power`)."""
    noise_good_return: float | None = None
    """The yearly return of the "good" strategy the power is asked about."""
    stage1_coins_shared: int | None = None
    """How many of stage 1's coins the second venue also lists, when stage 1's history was taken on
    those alone (rules from 2026-10-30.1)."""
    stage1_coins_total: int | None = None
    stage1_whole_history_reason: str | None = None
    """Why stage 1's whole history was used although the shared coins were asked for."""


def read_alert_percentile(path: str) -> float | None:
    """The owner's alarm percentile (`night/decay.yaml`); None while it is not set."""
    value = decay.load_config(Path(path))["alert_percentile"]
    return None if value is None else float(value)


def _rules(version: str, alert: float, total_metric: str = TOTAL_METRIC) -> RuleSet:
    return RuleSet(version=version, rules=[
        Rule(id=POSITIVE_RULE, stage=Stage.EDGE, metric=total_metric, comparator=Comparator.GT,
             threshold=0.0, fatal=True,
             rationale="The stage 2 window must have earned something after costs "
                       "(owner, 2026-10-27, «делай все»)."),
        Rule(id=CONSISTENCY_RULE, stage=Stage.EDGE, metric=CONSISTENCY_METRIC,
             comparator=Comparator.GT, threshold=alert, fatal=True,
             rationale="The share of same-length stage 1 segments (0-100) that were worse "
                       "must be above the owner's alarm percentile (night/decay.yaml): a "
                       "result in the bottom of stage 1's own history is unlike it."),
    ])


def _pct(value: float) -> str:
    return f"{value:+.1%}"


def review_stage2(ctx: RoutingContext, *, stage1_route: str,
                  load_history: Callable[[], Stage1History],
                  read_alert: Callable[[str], float | None] = read_alert_percentile,
                  ) -> tuple[RouteReview | None, Confirmation | None]:
    """The check, as a pure function of the run and of stage 1's history.

    (None, None): the check has nothing to replace -- no setting in the ruleset, the
    noise comparison did not fail, it has power, or another rule/the forward length/
    regime coverage keeps the run from passing anyway; the run's route stands as the
    rules gave it. Otherwise the review's route replaces it, with the numbers."""
    ruleset = ctx.ruleset
    setting, power_rule = ruleset.stage2_confirmation, ruleset.noise_power
    if (setting is None or power_rule is None or not ruleset.splits_fit_forward
            or not ctx.forward_judged):
        return None, None
    noise_metric, total_metric = power_rule.percentile_metric, setting.total_metric
    noise_rows = [r for r in ctx.rules_result.rows if r.metric == noise_metric]
    if not noise_rows or any(r.passed is not False for r in noise_rows):
        return None, None  # the comparison passed, or could not be computed: not replaced
    power = ctx.metrics.get(power_rule.power_metric)
    if power is None or power != power or power >= power_rule.min_power:
        return None, None  # not measured, or it has power: it decides as before

    # Every OTHER rule, the length of the forward test and regime coverage, as before.
    without_noise = ruleset.model_copy(update={
        "rules": [r for r in ruleset.rules if r.metric != noise_metric]})
    others = evaluate_rules(ctx.metrics, without_noise)
    without = decide_fit_forward_route(
        ruleset=without_noise, selection=ctx.selection_result, forward=others,
        metrics=dict(ctx.metrics),
        params_fixed_at=ctx.spec.params_fixed_at)
    if without.route not in PASSING:
        return None, None  # something other than the noise comparison keeps it waiting

    def not_applied(why: str, **numbers) -> tuple[RouteReview, Confirmation]:
        text = f"consistency check with stage 1 not applied: {why}"
        return (RouteReview(routing=dataclasses.replace(
            ctx.routing, reason=f"{ctx.routing.reason}; {text}")),
            Confirmation(status="not-applied", reason=why, rules_version=ruleset.version,
                         noise_power=float(power), noise_min_power=power_rule.min_power,
                         noise_good_return=power_rule.good_return, **numbers))

    if stage1_route not in STAGE1_CONFIRMS:
        return not_applied(f"stage 1 routed {stage1_route}: it must have decided first")
    alert = read_alert(setting.alert_percentile_from)
    if alert is None:
        return not_applied(f"the owner has not set `alert_percentile` in "
                           f"{setting.alert_percentile_from}")
    try:
        history = load_history()
    except Stage1HistoryMissing as exc:
        return not_applied(str(exc), alert_percentile=alert)
    except Exception as exc:  # noqa: BLE001 - the run must still be recorded, as it was before
        return not_applied(f"stage 1's history could not be computed: "
                           f"{type(exc).__name__}: {exc}", alert_percentile=alert)
    known = dict(alert_percentile=alert, stage1_trial_id=history.trial_id,
                 stage1_coins_shared=history.coins_shared, stage1_coins_total=history.coins_total,
                 stage1_whole_history_reason=history.whole_history_reason)
    # Stage 1's own power is not required: a stage 1 survivor PASSED its noise comparison
    # (an unpowered failure routes needs-forward, which is not in STAGE1_CONFIRMS), and low
    # power only lets good strategies fail -- it does not let empty ones pass a 99% bar.

    days = window_days_of(ctx.metrics, ctx.spec.data.interval)
    rule_total = ctx.metrics.get(total_metric)
    if days is None or rule_total is None or rule_total != rule_total:
        why = ("the stage 2 window has no total return to compare"
               if not ctx.metrics.get("cap_base_refused") else
               "the stage 2 book's base of percent is undefined (it takes deposits), so it has "
               "no total return on the strategy's own money to compare")
        return not_applied(why, **known)
    # Both sides on the strategy's own money when both have it and agree on how it grows.
    cap_total = ctx.metrics.get("cap_total_return")
    cap_flag = ctx.metrics.get("cap_reinvested")
    on_capital = (history.capital_return is not None
                  and isinstance(cap_total, int | float) and cap_total == cap_total
                  and isinstance(cap_flag, int | float) and bool(cap_flag) == history.reinvested)
    total = float(cap_total) if on_capital else float(rule_total)
    seg = segments_of(history.capital_return if on_capital else history.net_return, days,
                      additive=on_capital and not history.reinvested)
    if len(seg.totals) < decay.MIN_WINDOWS:
        return not_applied(
            f"stage 1's forward history ({seg.history_days} days) holds fewer than "
            f"{decay.MIN_WINDOWS} windows of the stage 2 length ({days} days)",
            window_days=days, total_return=float(total), on_capital=on_capital,
            stage1_noise_power=history.noise_power, **known)
    percentile = decay.percentile_of(float(total), list(seg.totals))
    assert percentile is not None  # the sample is not empty

    result = evaluate_rules({total_metric: float(rule_total), CONSISTENCY_METRIC: percentile},
                            _rules(ruleset.version, alert, total_metric))
    failed = set(result.failed_rule_ids)
    money = " (both on the strategy's own money)" if on_capital else ""
    where = (f"{percentile:.0f} of 100 same-length segments of stage 1's forward history "
             f"(trial {history.trial_id}, {seg.history_from} .. {seg.history_to}, "
             f"{len(seg.totals)} segments, {seg.independent} not overlapping) were worse "
             f"than {_pct(float(total))} over {days} days{money}; "
             f"the owner's alarm is the bottom {alert:g} of 100")
    base = dict(
        alert_percentile=alert, percentile=percentile, segments=len(seg.totals),
        independent_segments=seg.independent, window_days=days, total_return=float(total),
        on_capital=on_capital, stage1_trial_id=history.trial_id,
        stage1_from=seg.history_from.isoformat() if seg.history_from else None,
        stage1_to=seg.history_to.isoformat() if seg.history_to else None,
        stage1_coins_shared=history.coins_shared, stage1_coins_total=history.coins_total,
        stage1_whole_history_reason=history.whole_history_reason,
        noise_power=float(power), stage1_noise_power=history.noise_power,
        noise_min_power=power_rule.min_power, noise_good_return=power_rule.good_return,
        rules_version=ruleset.version)
    numbers = {
        "stage1_segment_percentile": percentile, "stage1_segments": float(len(seg.totals)),
        "stage1_independent_segments": float(seg.independent),
        "stage1_on_capital": 1.0 if on_capital else 0.0,
        "stage1_trial_id": float(history.trial_id), "stage1_alert_percentile": alert,
        "stage1_window_days": float(days)}
    if history.coins_shared is not None and history.coins_total is not None:
        numbers["stage1_coins_shared"] = float(history.coins_shared)
        numbers["stage1_coins_total"] = float(history.coins_total)
    if failed:
        what = []
        if POSITIVE_RULE in failed:
            what.append(
                f"its total return after costs, {_pct(float(rule_total)):s}, is not above zero")
        if CONSISTENCY_RULE in failed:
            what.append("it is inconsistent with stage 1: " + where)
        reason = "; ".join(what)
        status = "not-positive" if POSITIVE_RULE in failed else "inconsistent"
        review = RouteReview(
            routing=RoutingDecision(route="reject", reason=f"stage 2 does not confirm stage 1: "
                                    f"{reason}"),
            metrics=numbers, result=result, note=NOTE)
        return review, Confirmation(status=status, reason=reason, route="reject", **base)
    reason = (
        f"confirms stage 1 instead of the noise comparison, which cannot decide on this "
        f"window (a book of this shape truly earning {power_rule.good_return:.0%} a year would "
        f"pass it {power:.0%} of the time; it decides from {power_rule.min_power:.0%}): "
        f"{where}; total return is above zero; every other rule passed; "
        f"{without.reason} (rules {ruleset.version}). A consistency check, not a second proof")
    review = RouteReview(
        routing=RoutingDecision(route=without.route, reason=reason,
                                required_capital_usd=without.required_capital_usd),
        metrics=numbers, result=result, note=NOTE)
    return review, Confirmation(status="confirmed", reason=reason, route=without.route, **base)


class Stage2Reviewer:
    """The `review_route` of one stage 2 run; keeps what the check found in `outcome`."""

    def __init__(self, *, stage1_route: str, load_history: Callable[[], Stage1History],
                 read_alert: Callable[[str], float | None] = read_alert_percentile) -> None:
        self.stage1_route = stage1_route
        self.load_history = load_history
        self.read_alert = read_alert
        self.outcome: Confirmation | None = None

    def __call__(self, ctx: RoutingContext) -> RouteReview | None:
        review, self.outcome = review_stage2(
            ctx, stage1_route=self.stage1_route, load_history=self.load_history,
            read_alert=self.read_alert)
        return review


__all__ = ["CONSISTENCY_RULE", "Confirmation", "POSITIVE_RULE", "STAGE1_CONFIRMS", "Segments",
           "Stage1History", "Stage1HistoryMissing", "Stage2Reviewer", "load_stage1_history",
           "read_alert_percentile", "review_stage2", "segments_of", "window_days_of"]
