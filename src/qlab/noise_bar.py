"""The noise bar's dead band and its power gate (rules 2026-11-03.1, docs/NOISE_BAR.md).

Owner, 2026-11-03: «да, делай А+Б+Г». Two things about the bar `shape_aware_edge` (the place of a
run's yearly return, on the strategy's own money, among 200 random books of the same shape; the
line is 0.99):

**А, the dead band.** An idea passes at `pass_at` (0.99). Once it has passed, it loses the pass
only below `keep_above` (0.975). The rank among 200 books has an error of its own (about +-1.4
points at 0.99, one book being half a point), so a rank between the two numbers says nothing about
which side of the line the strategy is on; the line must not flip an idea's route on one new day of
data.

*"Once passed"* is exact (`previous_pass`): the idea's previous routed run on the same venue --
the latest earlier trial of the idea whose data came from the same source and which the bar judged
on its forward test (it has a decided verdict of the rule, not one of the selection period, which
the bar does not decide) -- passed the rule, and, when the ruleset says `no_power_no_pass`, the
bar had power on that run (a pass without power is not a pass, below). An idea with no such run has
not passed. A pass kept by the band is itself a pass, so the chain continues: a run at 0.98 after a
run at 0.995 keeps it, a next run at 0.98 keeps it again, and a run below 0.975 loses it.

The band moves the LINE used for the run, not the rule: the run is judged by the rules engine as
always, with the line of the rule set to `keep_above` when the previous run passed and the rank is
in the band (`with_kept_line`). The verdict row of the rule then shows that line and a note, and the
route's reason says so.

**Б, no power, no pass.** `NoisePower` measures how often the bar would pass a strategy of the run's
shape truly earning `good_return` a year (`noise_cap_power`). Below `min_power` the bar already
decided nothing by failing (a failure of a `wait` rule reads "too early to tell"); with
`no_power_no_pass` it decides nothing by passing either: the run's outcome is `needs-forward`
(`qlab.pipeline.evaluate.decide_fit_forward_route`). Stage 2's consistency check
(`qlab.pipeline.confirm`) is not changed: it replaces a failed noise comparison without power, and
is not asked about a pass.

Pure functions of a ruleset and metrics, apart from the one lookup in the registry
(`previous_pass`), which only reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qlab.rules.engine import EvaluationResult
from qlab.rules.schema import RuleSet

SELECTION_PREFIX = "selection period"
"""How the note of a verdict row of the selection period starts
(`pipeline.evaluate.SELECTION_NOTE`)."""


def _number(m: Mapping[str, object] | None, key: str) -> float | None:
    value = (m or {}).get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


def power_of(ruleset: RuleSet, metrics: Mapping[str, object] | None) -> float | None:
    """The measured power of the noise bar on a run (the metric the ruleset's `noise_power` names);
    None when it was not measured."""
    setting = ruleset.noise_power
    return None if setting is None else _number(metrics, setting.power_metric)


def powerless(ruleset: RuleSet, metrics: Mapping[str, object] | None) -> bool:
    """The noise bar cannot tell a good strategy from luck on this run: its measured power is below
    `min_power`, or it was not measured (rules 2026-10-24.2)."""
    setting = ruleset.noise_power
    if setting is None:
        return False
    power = power_of(ruleset, metrics)
    return power is None or power < setting.min_power


# --------------------------------------------------------------------------
# А: the dead band


@dataclass(frozen=True)
class PreviousPass:
    """What the bar said on the idea's previous routed run on this venue."""

    trial_id: int
    rank: float
    passed: bool
    """The rule passed on that run's forward test, and (when the ruleset gates passes by power) the
    bar had power on it."""
    powered: bool | None = None
    """Whether the bar had power on that run (None: not asked)."""


@dataclass(frozen=True)
class Band:
    """The dead band applied to one run."""

    rule_id: str
    rank: float
    pass_at: float
    keep_above: float
    previous: PreviousPass | None
    kept: bool
    """The previous run passed and the rank is in [keep_above, pass_at): the pass is kept."""

    @property
    def inside(self) -> bool:
        return self.keep_above <= self.rank < self.pass_at

    def note_ru(self) -> str:
        """«у черты: 0.985, проход держится до 0.975»: what the route's reason and the verdict row
        of the rule say when the band kept a pass."""
        return f"у черты: {self.rank:.3f}, проход держится до {self.keep_above:.3f}"


def previous_pass(session, spec, ruleset: RuleSet) -> PreviousPass | None:
    """The idea's previous routed run on the venue of `spec`, judged by the bar (module docstring);
    None when there is none."""
    from qlab.registry.models import Spec as SpecRow
    from qlab.registry.models import Trial, Verdict

    bar = ruleset.noise_bar
    if bar is None:
        return None
    venue = spec.data.source
    trials = (session.query(Trial, SpecRow).join(SpecRow, Trial.spec_id == SpecRow.id)
              .filter(SpecRow.idea_id == spec.idea_id, Trial.route.isnot(None),
                      Trial.metrics.isnot(None))
              .order_by(Trial.id.desc()))
    for trial, row in trials:
        if (row.data_requirements or {}).get("source") != venue:
            continue
        verdicts = (session.query(Verdict)
                    .filter(Verdict.trial_id == trial.id, Verdict.rule_id == bar.rule_id,
                            Verdict.passed.isnot(None), Verdict.value.isnot(None))
                    .order_by(Verdict.id.desc()).all())
        decided = [v for v in verdicts if not (v.note or "").startswith(SELECTION_PREFIX)]
        if not decided:
            continue  # the bar did not judge this run on a forward test: not an earlier verdict
        verdict = decided[0]
        powered = None
        if bar.no_power_no_pass:
            powered = not powerless(ruleset, trial.metrics)
        return PreviousPass(trial.id, float(verdict.value),
                            passed=bool(verdict.passed) and powered is not False, powered=powered)
    return None


def read_band(session, spec, ruleset: RuleSet, metrics: Mapping[str, object]) -> Band | None:
    """The band as it applies to a run with these metrics: None when the ruleset has none or the
    run has no rank. The registry is read only when the rank is inside the band."""
    bar = ruleset.noise_bar
    if bar is None or ruleset.noise_power is None:
        return None
    rank = _number(metrics, ruleset.noise_power.percentile_metric)
    if rank is None:
        return None
    band = Band(bar.rule_id, rank, bar.pass_at, bar.keep_above, None, kept=False)
    if not band.inside:
        return band
    previous = previous_pass(session, spec, ruleset)
    return Band(bar.rule_id, rank, bar.pass_at, bar.keep_above, previous,
                kept=previous is not None and previous.passed)


def with_kept_line(ruleset: RuleSet, band: Band) -> RuleSet:
    """The ruleset with the line of the noise rule moved down to the band's lower edge, for a run
    whose pass the band keeps."""
    rules = [r.model_copy(update={"threshold": band.keep_above}) if r.id == band.rule_id else r
             for r in ruleset.rules]
    return ruleset.model_copy(update={"rules": rules})


# --------------------------------------------------------------------------
# Б: no power, no pass


def gate_applies(ruleset: RuleSet, forward: EvaluationResult | None,
                 metrics: Mapping[str, object]) -> bool:
    """The noise rule passed on the judged forward test, but the bar has no power there: the pass
    is not counted (module docstring)."""
    bar = ruleset.noise_bar
    if bar is None or not bar.no_power_no_pass or forward is None:
        return False
    rows = [r for r in forward.rows if r.rule_id == bar.rule_id]
    return bool(rows) and all(r.passed is True for r in rows) and powerless(ruleset, metrics)


def power_note(ruleset: RuleSet, metrics: Mapping[str, object]) -> str:
    """The bar's power named in a route's reason: «noise bar power 83% (it decides from 80%)»."""
    setting = ruleset.noise_power
    if setting is None:
        return ""
    power = power_of(ruleset, metrics)
    if power is None:
        return "noise bar power not measured on this window"
    return f"noise bar power {power:.0%} (it decides from {setting.min_power:.0%})"


__all__ = ["Band", "PreviousPass", "gate_applies", "power_note", "power_of", "powerless",
           "previous_pass", "read_band", "with_kept_line"]
