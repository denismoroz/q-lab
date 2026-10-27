"""Tests for stage 2's confirmation by stage 1's history (rules 2026-10-27.1,
docs/TWO_STAGE.md): where the noise comparison cannot decide on the Hyperliquid window,
the stage 2 result is checked against same-length segments of stage 1's forward history.
Pure functions: no registry, no data, no agent."""

from __future__ import annotations

import dataclasses
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlab import decay
from qlab.diagnose import equity_from_returns
from qlab.pipeline import confirm
from qlab.pipeline.evaluate import RoutingContext, RoutingDecision
from qlab.rules.engine import evaluate as evaluate_rules
from qlab.rules.loader import DEFAULT_RULES_DIR, load
from qlab.rules.schema import (
    Comparator,
    FitPeriodUse,
    HoldoutShare,
    NoisePower,
    Rule,
    RuleSet,
    ShortForwardUse,
    Stage,
    Stage2Confirmation,
)

ALERT = 7.0
NOISE = Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="noise_return_percentile",
             comparator=Comparator.GE, threshold=0.99, fatal=True,
             fit_period=FitPeriodUse.INFORMATIONAL, on_short_forward=ShortForwardUse.WAIT)
EDGE = Rule(id="net_edge_positive", stage=Stage.EDGE, metric="ann_return_net",
            comparator=Comparator.GT, threshold=0.0, fatal=True)
CAPITAL = Rule(id="capital_fit", stage=Stage.PREFLIGHT, metric="min_capital_usd",
               comparator=Comparator.LE, threshold=1_000_000, fatal=True)
SETTING = Stage2Confirmation(alert_percentile_from="night/decay.yaml", rationale="test")


def _ruleset(*rules: Rule, confirmation: bool = True) -> RuleSet:
    return RuleSet(
        version="2026-10-27.1", rules=list(rules or (NOISE, EDGE, CAPITAL)),
        holdout_share=HoldoutShare(share=1 / 3, rationale="t"),
        noise_power=NoisePower(good_return=0.07, min_power=0.80, rationale="t"),
        stage2_confirmation=SETTING if confirmation else None)


def _metrics(**over: float) -> dict[str, float]:
    base = {"ann_return_net": 0.445, "total_return_net": 0.50, "sharpe_net": 1.34,
            "n_periods": 100.0, "forward_days": 100.0, "min_capital_usd": 500.0,
            "noise_return_percentile": 0.935, "noise_power": 0.06}
    base.update(over)
    return base


def _context(metrics: dict[str, float], ruleset: RuleSet | None = None,
             capital: float = 3000.0) -> RoutingContext:
    ruleset = ruleset or _ruleset()
    result = evaluate_rules(metrics, ruleset)
    routing = RoutingDecision(route="needs-forward",
                              reason="too early to tell: 100 days fail only shape_aware_edge")
    spec = types.SimpleNamespace(params_fixed_at=date(2025, 1, 1),
                                 data=types.SimpleNamespace(interval="1d"))
    return RoutingContext(
        spec=spec, ruleset=ruleset, routing=routing, metrics=metrics, rules_result=result,
        selection_result=None, forward_judged=True, deployable_capital_usd=capital,
        range_start=date(2025, 1, 1), range_end=date(2025, 4, 10))


def _stage1(days: int = 1000, drift: float = 0.002, noise: float = 0.01, seed: int = 4,
            power: float | None = 0.95, trial_id: int = 11) -> confirm.Stage1History:
    index = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    returns = np.random.default_rng(seed).normal(drift, noise, days)
    return confirm.Stage1History(trial_id, pd.Series(returns, index=index), power)


def _review(ctx: RoutingContext, history: confirm.Stage1History | Exception,
            stage1_route: str = "paper", alert: float | None = ALERT):
    def load_history() -> confirm.Stage1History:
        if isinstance(history, Exception):
            raise history
        return history

    return confirm.review_stage2(ctx, stage1_route=stage1_route, load_history=load_history,
                                 read_alert=lambda path: alert)


def _median_total(history: confirm.Stage1History, days: int) -> float:
    return float(np.median(confirm.segments_of(history.net_return, days).totals))


# ---- the cases the owner named ----------------------------------------------------------


def test_a_noise_comparison_with_power_decides_as_before() -> None:
    ctx = _context(_metrics(noise_power=0.90))
    assert _review(ctx, _stage1()) == (None, None)


def test_a_passed_noise_comparison_is_left_alone() -> None:
    ctx = _context(_metrics(noise_return_percentile=0.995, noise_power=0.05))
    assert _review(ctx, _stage1()) == (None, None)


def test_an_unmeasured_power_leaves_todays_route() -> None:
    metrics = _metrics()
    del metrics["noise_power"]
    assert _review(_context(metrics), _stage1()) == (None, None)


def test_no_power_and_a_consistent_result_confirms_stage_1() -> None:
    history = _stage1()
    total = _median_total(history, 100)
    review, found = _review(_context(_metrics(total_return_net=total)), history)

    assert review.routing.route == "paper" and found.route == "paper"
    assert found.status == "confirmed" and found.rules_version == "2026-10-27.1"
    assert 40 < found.percentile < 60  # the median segment sits in the middle
    assert found.segments == len(confirm.segments_of(history.net_return, 100).totals)
    assert found.stage1_trial_id == 11 and found.window_days == 100
    assert found.alert_percentile == ALERT and found.noise_power == 0.06
    assert "consistency check, not a second proof" in review.routing.reason
    assert "same-length segments of stage 1's forward history" in review.routing.reason
    # the numbers go on the trial, the decision on the verdicts, by the rules engine
    assert review.metrics["stage1_segment_percentile"] == pytest.approx(found.percentile)
    assert review.metrics["stage1_trial_id"] == 11.0
    rows = {r.rule_id: r for r in review.result.rows}
    assert set(rows) == {confirm.POSITIVE_RULE, confirm.CONSISTENCY_RULE}
    assert all(r.passed and r.rules_version == "2026-10-27.1" for r in rows.values())
    assert rows[confirm.CONSISTENCY_RULE].threshold == ALERT


def test_no_power_and_a_result_in_the_bottom_7_of_100_rejects() -> None:
    history = _stage1(drift=0.01, noise=0.002)  # every 100-day segment earns far more
    review, found = _review(_context(_metrics(total_return_net=0.02)), history)

    assert review.routing.route == "reject" and found.route == "reject"
    assert found.status == "inconsistent" and found.percentile <= ALERT
    assert "inconsistent with stage 1" in review.routing.reason
    failed = [r.rule_id for r in review.result.rows if r.passed is False]
    assert failed == [confirm.CONSISTENCY_RULE]  # the failed verdict carries the data range


def test_the_alarm_is_the_owners_inclusive_edge() -> None:
    """At exactly the alarm percentile a result counts as unlike its history (the paper
    check's `alert_label`): 7 of 100 worse is already inside the bottom 7."""
    history = _stage1()
    segments = confirm.segments_of(history.net_return, 100).totals
    total = float(np.quantile(segments, 0.07))
    percentile = decay.percentile_of(total, list(segments))
    review, found = _review(_context(_metrics(total_return_net=total)), history,
                            alert=percentile)
    assert found.status == "inconsistent"
    assert decay.alert_label(percentile, percentile) == "below"
    just_above = _review(_context(_metrics(total_return_net=total)), history,
                         alert=percentile - 0.01)[1]
    assert just_above.status == "confirmed"


def test_a_total_return_not_above_zero_rejects_even_if_stage_1_looked_alike() -> None:
    history = _stage1(drift=-0.0005, noise=0.01)  # a history where a loss is ordinary
    ruleset = _ruleset(NOISE, CAPITAL)  # no yearly-return rule: the total is what is checked
    total = _median_total(history, 100)
    assert total < 0
    review, found = _review(
        _context(_metrics(total_return_net=total, ann_return_net=-0.1), ruleset), history)
    assert review.routing.route == "reject" and found.status == "not-positive"
    assert "not above zero" in found.reason
    assert any(r.rule_id == confirm.POSITIVE_RULE and r.passed is False
               for r in review.result.rows)


def test_a_stage_1_that_is_still_too_early_confirms_nothing() -> None:
    history = _stage1()
    ctx = _context(_metrics(total_return_net=_median_total(history, 100)))
    review, found = _review(ctx, history, stage1_route="needs-forward")
    assert review.routing.route == "needs-forward" and found.status == "not-applied"
    assert found.route is None and "it must have decided first" in found.reason
    assert review.metrics == {} and review.result is None


def test_missing_stage_1_history_keeps_todays_route_and_says_why() -> None:
    error = confirm.Stage1HistoryMissing("stage 1's data snapshot snap-1 cannot be read: gone")
    review, found = _review(_context(_metrics()), error)
    assert review.routing.route == "needs-forward" and found.status == "not-applied"
    assert review.routing.reason.startswith("too early to tell: 100 days fail only")
    assert ("consistency check with stage 1 not applied: stage 1's data snapshot snap-1 cannot "
            "be read: gone") in review.routing.reason
    assert found.route is None and review.result is None


def test_a_crash_computing_stage_1_history_is_a_reason_not_a_crash() -> None:
    review, found = _review(_context(_metrics()), ValueError("boom"))
    assert review.routing.route == "needs-forward"
    assert "stage 1's history could not be computed: ValueError: boom" in found.reason


@pytest.mark.parametrize("power,expect", [(None, "was not recorded"), (0.5, "had power 50%")])
def test_a_stage_1_whose_own_noise_comparison_was_unpowered_is_not_a_proof(
        power: float | None, expect: str) -> None:
    review, found = _review(_context(_metrics()), _stage1(power=power))
    assert found.status == "not-applied" and expect in found.reason
    assert review.routing.route == "needs-forward"


def test_without_the_owners_alarm_nothing_is_confirmed() -> None:
    review, found = _review(_context(_metrics()), _stage1(), alert=None)
    assert found.status == "not-applied" and "has not set `alert_percentile`" in found.reason


def test_a_stage_1_history_shorter_than_the_stage_2_window_has_nothing_to_compare() -> None:
    review, found = _review(_context(_metrics(n_periods=100.0)), _stage1(days=80))
    assert found.status == "not-applied" and "holds fewer than 2 windows" in found.reason
    assert review.routing.route == "needs-forward"


def test_another_failed_rule_keeps_todays_route() -> None:
    ctx = _context(_metrics(ann_return_net=-0.05, total_return_net=-0.01))
    assert _review(ctx, _stage1()) == (None, None)  # net_edge_positive fails: not a pass


def test_a_book_too_dear_for_the_capital_goes_to_the_shelf_not_paper() -> None:
    history = _stage1()
    ctx = _context(_metrics(total_return_net=_median_total(history, 100), min_capital_usd=9e5),
                   capital=3000.0)
    review, found = _review(ctx, history)
    assert review.routing.route == "shelf" and review.routing.required_capital_usd == 9e5


def test_a_forward_test_that_is_too_short_keeps_waiting() -> None:
    ruleset = _ruleset()
    # a selection period of 400 days makes the forward test need 200 days; it has 100
    metrics = _metrics(forward_days=100.0, forward_days_needed=200.0, fit_window_days=400.0,
                       selection_days=400.0)
    ctx = _context(metrics, ruleset)
    selection = evaluate_rules({"min_capital_usd": 500.0, "ann_return_net": 0.2}, ruleset)
    ctx = dataclasses.replace(ctx, selection_result=selection)
    assert _review(ctx, _stage1()) == (None, None)


def test_a_ruleset_without_the_setting_changes_nothing() -> None:
    ctx = _context(_metrics(), _ruleset(confirmation=False))
    assert _review(ctx, _stage1()) == (None, None)


# ---- the segments -------------------------------------------------------------------------


def test_segments_are_the_paper_checks_rolling_windows() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    returns = pd.Series(0.01, index=index)
    found = confirm.segments_of(returns, 10)

    closes = decay.daily_closes(equity_from_returns(returns))
    assert found.totals == tuple(w.total for w in decay.rolling_windows(closes, 10))
    assert len(found.totals) == 30 + 1 - 10 and found.history_days == 30
    assert found.independent == 3  # 30 days hold three 10-day segments side by side
    assert found.totals[0] == pytest.approx(1.01 ** 10 - 1)


def test_the_window_is_the_periods_times_the_bar() -> None:
    assert confirm.window_days_of({"n_periods": 627.0}, "1d") == 627
    assert confirm.window_days_of({"n_periods": 42.0}, "4h") == 7
    assert confirm.window_days_of({}, "1d") is None
    assert confirm.window_days_of({"n_periods": 5.0}, "bogus") is None


# ---- stage 1's history is read back from its own trial's snapshot ---------------------------


class _Session:
    def __init__(self, trial) -> None:
        self.trial = trial

    def get(self, model, key):
        return self.trial if key == 7 else None


def _stage1_world(monkeypatch, *, fixed_at: date | None = date(2026, 2, 1)):
    from qlab.data.panel import MarketPanel

    index = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    prices = pd.DataFrame({"BTC": 100.0 * 1.01 ** np.arange(100)}, index=index)
    zero = pd.DataFrame(0.0, index=index, columns=["BTC"])
    panel = MarketPanel(
        snapshot_id="snap-1", prices=prices, funding=zero,
        tradeable=pd.DataFrame(True, index=index, columns=["BTC"]), volume=zero + 1.0,
        high=prices, meta={})

    class Strategy:
        def target_weights(self, panel, params):
            return pd.DataFrame(1.0, index=panel.prices.index, columns=["BTC"])

    import qlab.data.snapshot as snapshot
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(snapshot, "load_snapshot", lambda sid, session=None: panel)
    monkeypatch.setattr(evaluate, "resolve_strategy", lambda ref: Strategy())
    spec = types.SimpleNamespace(
        idea_id="x", data=object(), required_instruments=None, code_ref="x:y", params={},
        params_fixed_at=fixed_at,
        costs=types.SimpleNamespace(taker_fee_bps=0.0, slippage_bps=0.0))
    trial = types.SimpleNamespace(snapshot_id="snap-1", metrics={"noise_power": 0.93})
    return spec, _Session(trial)


def test_stage_1s_history_is_its_strategy_on_its_trials_snapshot_from_the_fix_date(
        monkeypatch) -> None:
    spec, session = _stage1_world(monkeypatch)
    history = confirm.load_stage1_history(session, spec, 7)
    assert history.trial_id == 7 and history.noise_power == 0.93
    assert history.net_return.index[0] == pd.Timestamp("2026-02-01", tz="UTC")  # forward only
    assert history.net_return.index[-1] < pd.Timestamp("2026-04-10", tz="UTC")


@pytest.mark.parametrize("trial_id,fixed_at,expect", [
    (None, date(2026, 2, 1), "no stage 1 trial is recorded"),
    (8, date(2026, 2, 1), "stage 1 trial 8 is not in the registry"),
    (7, None, "no forward history"),
])
def test_stage_1s_history_that_cannot_be_read_says_why(
        monkeypatch, trial_id, fixed_at, expect) -> None:
    spec, session = _stage1_world(monkeypatch, fixed_at=fixed_at)
    with pytest.raises(confirm.Stage1HistoryMissing, match=expect):
        confirm.load_stage1_history(session, spec, trial_id)


def test_a_snapshot_gone_from_disk_is_a_missing_history_not_a_crash(monkeypatch) -> None:
    spec, session = _stage1_world(monkeypatch)
    import qlab.data.snapshot as snapshot

    def gone(snapshot_id, session=None):
        raise FileNotFoundError("prices.parquet")

    monkeypatch.setattr(snapshot, "load_snapshot", gone)
    with pytest.raises(confirm.Stage1HistoryMissing, match="snap-1 cannot be read"):
        confirm.load_stage1_history(session, spec, 7)


# ---- the shipped rules --------------------------------------------------------------------


def test_the_shipped_rules_read_the_owners_alarm_from_the_decay_config() -> None:
    ruleset = load("2026-10-27.1", rules_dir=DEFAULT_RULES_DIR)
    assert ruleset.based_on == "2026-10-24.2"
    assert ruleset.noise_power.min_power == 0.80 and ruleset.holdout_share is not None
    assert ruleset.stage2_confirmation.alert_percentile_from == "night/decay.yaml"
    assert confirm.read_alert_percentile("night/decay.yaml") == 7.0  # not copied into the rules
    assert "alert_percentile:" not in (DEFAULT_RULES_DIR / "2026-10-27.1.yaml").read_text()


def test_no_older_ruleset_has_the_setting() -> None:
    assert load("2026-10-24.2", rules_dir=DEFAULT_RULES_DIR).stage2_confirmation is None


def test_the_confirmation_module_calls_no_agent() -> None:
    source = Path(confirm.__file__).read_text(encoding="utf-8")
    assert "run_agent" not in source and "qlab.agents" not in source
    assert "qlab.budget" not in source
