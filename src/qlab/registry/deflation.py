"""Deflation: how much of a result is explained by the number of attempts.

`docs/PLAN.md` and `CLAUDE.md` have declared since the first commit that every
run is written to the trial ledger "так что дефляция считается по фактическому
числу испытаний". Until now nothing computed it. The ledger was honest and
unread, which is the same as not having one: T21 produced a strategy routed to
`paper` at Sharpe 1.042 with no statement anywhere of how many looks at the data
that number stood on (`docs/XSMOM_T21.md`, T27 item 4).

**The question this module answers.** A Sharpe of 1.0 found on the first try and
a Sharpe of 1.0 found on the fiftieth are not the same evidence. Searching over
variants raises the best result you will see even when nothing has any edge at
all, and the amount it raises it by is computable from the number of trials and
the spread of their results. Comparing an observed Sharpe against that inflated
benchmark, rather than against zero, is deflation.

The estimators are Bailey & Lopez de Prado, "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality" (Journal of Portfolio
Management, 2014). They are taken from the literature rather than chosen here,
which matters: this project forbids inventing thresholds, and every input below
(`n_trials`, the spread of trial Sharpes, the sample length, the skew and
kurtosis of the returns) is read off our own ledger and our own backtests.

**What this module assumes, and where it is wrong in our favour.** The expected
maximum assumes the trials are independent. Ours are emphatically not: three
XSMOM variants share a signal, a universe and a window, and their results move
together. Correlated trials search less of the space than independent ones do, so
the true effective count is BELOW `n_trials`, the benchmark computed here is too
high, and the deflated Sharpe that comes out is too harsh. That is the safe
direction to be wrong in, and it is the reason this is usable now rather than
after someone works out an effective-trials correction. It is not a reason to
treat a passing DSR as generous -- it is a reason to treat a failing one as
settled.

**What counts as a trial is a judgement, so this module refuses to make it.**
`family_trials` takes an explicit `scope` with no default. Counting only the
trials filed under one `idea` id says that trying the same strategy at a
different book width is a different strategy -- which is exactly how XSMOM's
three variants ended up looking like one attempt each. Counting every trial
sharing a `driver` says they are all looks at the same edge, which is the
honest reading for deflation and the conservative one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from qlab.registry.models import Idea, Spec, Trial

# Euler-Mascheroni constant, as it appears in the expected-maximum estimator.
_EULER_MASCHERONI = 0.5772156649015329

_NORMAL = NormalDist()

Scope = Literal["idea", "driver"]


class DeflationError(ValueError):
    """Raised when deflation is asked for on inputs that cannot support it.

    Deliberately an error rather than a NaN or a sentinel: a caller that gets a
    number back must be able to trust that a real calculation happened, the same
    contract `qlab.harness.accrual` enforces for a run without accrual.
    """


@dataclass(frozen=True, slots=True)
class TrialCount:
    """How many trials a result stands on, and what was counted as one."""

    scope: Scope
    key: str
    n_trials: int
    n_ok: int
    idea_ids: tuple[str, ...]


def family_trials(session: Session, idea_id: str, *, scope: Scope) -> TrialCount:
    """Count the trials one result stands on.

    `scope="idea"` counts only trials whose spec names this idea. `scope="driver"`
    counts every trial for every idea sharing this idea's driver -- the reading
    that treats variants of one edge as repeated looks at the same data, which is
    what deflation is about. There is no default: see the module docstring.

    Errored trials are counted in `n_trials` and excluded from `n_ok`. Both
    numbers are reported because they answer different questions: a run that
    crashed still consumed a look at the data if it got far enough to compute
    anything, while only successful runs contribute Sharpes to the spread.
    """
    idea = session.get(Idea, idea_id)
    if idea is None:
        raise DeflationError(f"unknown idea {idea_id!r}")

    if scope == "idea":
        idea_ids = [idea_id]
        key = idea_id
    elif scope == "driver":
        if idea.driver_id is None:
            raise DeflationError(
                f"idea {idea_id!r} has no driver, so driver scope has no meaning; "
                "either give it one or ask for scope='idea' and say so in the report"
            )
        idea_ids = list(
            session.scalars(select(Idea.id).where(Idea.driver_id == idea.driver_id)).all()
        )
        key = idea.driver_id
    else:  # pragma: no cover - Literal keeps this unreachable from typed callers
        raise DeflationError(f"unknown scope {scope!r}")

    rows = session.execute(
        select(Trial.status).join(Spec, Spec.id == Trial.spec_id).where(Spec.idea_id.in_(idea_ids))
    ).all()
    n_trials = len(rows)
    n_ok = sum(1 for (status,) in rows if getattr(status, "value", str(status)) == "ok")
    return TrialCount(
        scope=scope, key=key, n_trials=n_trials, n_ok=n_ok, idea_ids=tuple(idea_ids)
    )


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """The best Sharpe a search over `n_trials` produces when nothing has edge.

    Bailey & Lopez de Prado's estimator of the expected maximum of `n_trials`
    draws from a normal with mean zero and standard deviation `sharpe_std`:

        E[max] = sharpe_std * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ]

    with `g` the Euler-Mascheroni constant and `Z` the inverse standard normal
    CDF. Units follow `sharpe_std`: pass per-observation Sharpes and you get a
    per-observation benchmark back. `qlab.harness.metrics` reports
    `sharpe_net` ANNUALISED, so divide by sqrt(periods per year) first --
    `deflated_sharpe` below does this for you and is the safer entry point.

    Rises with `n_trials` without bound, slowly: this is the whole point. Two
    trials barely move it; fifty move it a lot.
    """
    if n_trials < 2:
        raise DeflationError(
            f"expected_max_sharpe needs at least 2 trials, got {n_trials}: with a single "
            "trial there is no maximum to take and no selection to correct for"
        )
    if not math.isfinite(sharpe_std) or sharpe_std <= 0.0:
        raise DeflationError(
            f"sharpe_std must be positive and finite, got {sharpe_std!r}: a zero spread "
            "would claim every trial returned exactly the same Sharpe, which is a bug "
            "in the caller's trial set rather than a strategy with no selection bias"
        )
    n = float(n_trials)
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    return sharpe_std * ((1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second)


def deflated_sharpe(
    *,
    sharpe_net: float,
    n_trials: int,
    sharpe_std_net: float,
    n_periods: int,
    periods_per_year: float,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability that the observed Sharpe is not just the best of the search.

    Returns a probability in [0, 1]. A value near 1 means the result survives
    the number of attempts behind it; near 0 means a search over that many
    variants would be expected to produce something this good with no edge
    present at all.

        DSR = Z( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    `SR` and `SR*` are PER-OBSERVATION; `sharpe_net` and `sharpe_std_net` are
    taken in the annualised units `qlab.harness.metrics` produces and converted
    here, so callers never have to remember which convention they are holding.
    `kurtosis` is NON-EXCESS (3.0 for a normal distribution) -- pandas' `.kurt()`
    returns EXCESS kurtosis, so add 3.0 to it before passing it in. Getting this
    backwards inflates the denominator and quietly makes every result look more
    robust than it is.

    The denominator is where non-normality enters: negative skew and fat tails
    both make a given Sharpe less trustworthy, which is the correction that
    matters for strategies whose returns come from a handful of days.
    """
    if n_periods < 2:
        raise DeflationError(f"n_periods must be at least 2, got {n_periods}")
    if periods_per_year <= 0:
        raise DeflationError(f"periods_per_year must be positive, got {periods_per_year}")

    scale = math.sqrt(periods_per_year)
    sr = sharpe_net / scale
    sr_star = expected_max_sharpe(n_trials, sharpe_std_net / scale)

    variance_term = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if variance_term <= 0.0:
        raise DeflationError(
            f"non-normality term is non-positive ({variance_term:.6g}); the skew/kurtosis "
            "pair cannot describe a real return series, so the inputs are wrong rather "
            "than the strategy being infinitely good"
        )

    z = (sr - sr_star) * math.sqrt(n_periods - 1.0) / math.sqrt(variance_term)
    return _NORMAL.cdf(z)
