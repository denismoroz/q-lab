"""Re-running a stored trial for the anomaly work (docs/ANOMALIES.md).

A trial stores numbers, not the book. To look inside a run that looks too good -- which day,
which coin, how much -- the stored spec is run again on the stored data snapshot, CPU only:
no network, no agent, no trial, no verdict, no registry write. `forward_series.replay` does
the same for the forward test's chart; this module cuts the same rows (`judged_rows`) but
keeps the whole book, so the result can be taken apart.

* `load_run` -- the stored spec, snapshot and weights of a trial, and the rows it was judged on;
* `decompose` -- the backtest taken apart by instrument: for every decision bar and every
  instrument the weight, the price move, and the price, funding and cost parts of the net
  return. It repeats `qlab.harness.run.run_backtest` step by step and says whether its sum
  equals what `run_backtest` returns (`Decomposition.reconciles`), so a drift between the two
  is visible instead of silent;
* `Rule` and `apply_rule` -- one declared change of the weights for a counterfactual ("what if
  the coin-days under this volume were not held"). Only ever removes exposure.

Nothing here decides anything: the numbers are for the detector (`qlab.anomalies.flags`) and
for the investigator's read-only toolkit (`qlab.anomalies.toolkit`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from qlab.harness.costs import CostModel
from qlab.harness.gaps import held_through, holdable_gaps, marked_through
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest
from qlab.harness.strategy import ZERO_WEIGHT_TOL, capital_setup, own_costs

RECONCILE_TOLERANCE = 1e-9
"""How far the taken-apart sum may sit from `run_backtest`'s net return per bar."""


class RunUnavailable(Exception):  # noqa: N818 - a stated reason, not a failure
    """The trial cannot be re-run, and why (no numbers, no snapshot, the spec file is gone)."""


@dataclass
class StoredRun:
    """A trial's spec, data and book, ready to be backtested again on its judged rows."""

    trial_id: int
    idea_id: str
    spec: Any
    panel: MarketPanel
    weights: pd.DataFrame
    judged: tuple[int, int]
    """First and last row of the panel the verdict was rendered on."""
    costs: CostModel
    extra_cost: pd.Series | None
    route: str | None = None
    metrics: dict = field(default_factory=dict)
    capital: Any = None
    """The strategy's base and declared reinvest choice (`qlab.harness.strategy.CapitalSetup`),
    so the run can also be read on the strategy's own money (docs/CAPITAL_BASE.md); None reads
    it as the stand always did."""

    @property
    def interval(self) -> str:
        return self.spec.data.interval

    @property
    def venue(self) -> str:
        return self.spec.data.source

    @property
    def judged_for(self) -> str:
        """The venue this run is judged for: where its orders would go (`execution_venue`), else the
        venue of its data. Since the venue split (docs/VENUE_SPLIT.md) a run on Binance's data is
        judged for Binance, not as a proxy for Hyperliquid."""
        return getattr(self.spec, "execution_venue", None) or self.spec.data.source

    def part(self, weights: pd.DataFrame | None = None) -> tuple[MarketPanel, pd.DataFrame]:
        """The judged rows of the panel and of `weights` (the stored book by default)."""
        from qlab.pipeline.evaluate import _slice_rows

        lo, hi = self.judged
        w = self.weights if weights is None else weights
        return _slice_rows(self.panel, lo, hi), w.iloc[lo:hi + 1]

    def backtest(self, weights: pd.DataFrame | None = None) -> RunResult:
        """The judged rows run through the harness, with `weights` or the stored book."""
        part_panel, part_weights = self.part(weights)
        capital = {} if self.capital is None else self.capital.kwargs
        return run_backtest(part_panel, part_weights, self.costs, part_panel.funding,
                            extra_cost=self.extra_cost, **capital)


def load_run(session: Session, trial_id: int, *, spec_root: Path | None = None) -> StoredRun:
    """Re-create the trial's book from its stored spec and snapshot. Raises `RunUnavailable`
    with the reason when that cannot be done. Reads the registry; writes nothing."""
    from qlab.data.snapshot import load_snapshot
    from qlab.forward_series import _rules_version, spec_file_for
    from qlab.pipeline.evaluate import (
        complete_book_window,
        cut_truncated_history,
        judged_rows,
        resolve_strategy,
    )
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Spec, Trial
    from qlab.rules import loader

    trial = session.get(Trial, trial_id)
    if trial is None:
        raise RunUnavailable(f"no trial {trial_id} in the registry")
    if trial.metrics is None or trial.snapshot_id is None:
        raise RunUnavailable("the trial has no numbers or no data snapshot: nothing to re-run")
    spec_row = session.get(Spec, trial.spec_id)
    path = spec_file_for(session, trial.spec_id, spec_root)
    if spec_row is None or path is None:
        raise RunUnavailable("the spec file this trial ran with is not found or has changed "
                             "since: the run cannot be honestly reproduced")
    spec = load_spec(path)
    version = _rules_version(session, trial_id)
    try:
        ruleset = loader.load(version) if version else loader.load_latest()
    except Exception:  # noqa: BLE001 - a retired ruleset file: the latest cuts rows the same way
        ruleset = loader.load_latest()
    try:
        panel = load_snapshot(trial.snapshot_id, session=session)
    except (ValueError, OSError) as exc:
        raise RunUnavailable(f"the data this trial ran on is not on disk: {exc}") from exc
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    rows = judged_rows(spec, panel, coverage, ruleset)
    try:
        strategy = resolve_strategy(spec.code_ref)
        weights = strategy.target_weights(panel, spec.params)
        extra = own_costs(strategy, panel, spec.params)
        capital = capital_setup(strategy, panel, spec.params)
    except Exception as exc:  # noqa: BLE001 - strategy code is arbitrary
        raise RunUnavailable(f"the run could not be repeated: {type(exc).__name__}: {exc}") from exc
    return StoredRun(
        trial_id=trial_id, idea_id=spec.idea_id, spec=spec, panel=panel, weights=weights,
        judged=rows.judged,
        costs=CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                        slippage_bps=spec.costs.slippage_bps),
        extra_cost=extra, route=None if trial.route is None else str(trial.route.value),
        metrics=dict(trial.metrics), capital=capital)


# --------------------------------------------------------------------------
# The backtest taken apart


@dataclass
class Decomposition:
    """`run_backtest` by instrument, for the held instruments only (an instrument the book never
    holds in the judged rows contributes nothing and is left out).

    Every frame is indexed by decision bar and has one column per held instrument.
    `net = price + funding - cost`; the book's net return per bar is `net.sum(axis=1)` minus the
    `extra_cost` the strategy pays as a whole."""

    weights: pd.DataFrame
    move: pd.DataFrame
    """The instrument's price return over the bar after the decision (NaN: no price)."""
    price: pd.DataFrame
    funding: pd.DataFrame
    cost: pd.DataFrame
    extra_cost: pd.Series
    prices: pd.DataFrame
    """The instrument's price at the decision bar."""
    volume_usd: pd.DataFrame
    """Traded volume of the bar in dollars (`volume * price`); NaN where the volume is unknown."""
    tradeable: pd.DataFrame
    first_priced: pd.Series
    """Per instrument: the first bar of the WHOLE panel that has a price (its listing, for a coin
    listed after the panel starts)."""
    last_priced: pd.Series
    """Per instrument: the last bar of the WHOLE panel that has a price (not only the judged
    rows), and `panel_end` the panel's last bar -- a coin whose last price is well before the
    panel's end stopped trading."""
    panel_end: pd.Timestamp
    reconciles: bool
    max_gap: float
    """Largest per-bar difference to `run_backtest`'s net return."""

    @property
    def net(self) -> pd.DataFrame:
        return self.price + self.funding - self.cost

    @property
    def book_net(self) -> pd.Series:
        return self.net.sum(axis=1) - self.extra_cost

    @property
    def positions(self) -> pd.Series:
        """How many instruments are held at each decision bar."""
        return (self.weights.abs() > ZERO_WEIGHT_TOL).sum(axis=1)

    @property
    def gross(self) -> pd.Series:
        """Sum of the absolute weights at each decision bar."""
        return self.weights.abs().sum(axis=1)


def decompose(run: StoredRun, weights: pd.DataFrame | None = None) -> Decomposition:
    """Take the judged rows of `run` (with `weights` instead of the stored book, for a
    counterfactual) apart by instrument. See `Decomposition`."""
    part_panel, part_weights = run.part(weights)
    held = [c for c in part_weights.columns if (part_weights[c].abs() > ZERO_WEIGHT_TOL).any()]
    prices_all = part_panel.prices
    keep = prices_all.index[:-1]
    no_funding = tuple(part_panel.meta.get("no_funding_instruments", ()))

    prices = prices_all[held]
    funding_all = part_panel.funding[held]
    gaps = holdable_gaps(prices, funding_all, [c for c in no_funding if c in held])
    w = held_through(part_weights[held], gaps)
    marked = marked_through(prices, gaps)

    move = marked.pct_change().shift(-1).loc[keep]
    w_kept = w.loc[keep]
    prev = w.shift(1).fillna(0.0).loc[keep]
    cost = (w_kept - prev).abs() * (run.costs.total_bps / 1e4)
    price = (w_kept * move).fillna(0.0)
    funding_fwd = funding_all.shift(-1).loc[keep]
    funding = -(w_kept * funding_fwd.fillna(0.0))
    extra = (run.extra_cost.reindex(keep).fillna(0.0) if run.extra_cost is not None
             else pd.Series(0.0, index=keep))

    volume_usd = (part_panel.volume[held] * prices).loc[keep]
    first_priced = run.panel.prices[held].apply(
        lambda col: col.first_valid_index() if col.notna().any() else pd.NaT)
    last_priced = run.panel.prices[held].apply(
        lambda col: col.last_valid_index() if col.notna().any() else pd.NaT)
    dec = Decomposition(
        weights=w_kept, move=move, price=price, funding=funding, cost=cost, extra_cost=extra,
        prices=prices.loc[keep], volume_usd=volume_usd,
        tradeable=part_panel.tradeable[held].loc[keep],
        first_priced=first_priced, last_priced=last_priced,
        panel_end=run.panel.prices.index[-1], reconciles=True, max_gap=0.0)
    expected = run.backtest(weights).net_return
    gap = float((dec.book_net - expected.reindex(dec.book_net.index)).abs().max()) \
        if len(expected) else 0.0
    dec.max_gap = gap
    dec.reconciles = bool(np.isfinite(gap) and gap <= RECONCILE_TOLERANCE)
    return dec


def daily(series: pd.Series | pd.DataFrame, how: str = "sum") -> pd.Series | pd.DataFrame:
    """Bars grouped into UTC days. `how`: "sum" (arithmetic contributions), "compound"
    (returns), "mean" or "last"."""
    days = pd.DatetimeIndex(series.index).tz_convert("UTC").normalize()
    grouped = series.groupby(days)
    if how == "compound":
        return (1.0 + series).groupby(days).prod() - 1.0
    return getattr(grouped, how)()


# --------------------------------------------------------------------------
# Counterfactual rules


@dataclass(frozen=True)
class Rule:
    """One declared change of the book. Exactly one of the fields is set.

    * `min_volume_usd` -- coin-days whose traded volume (dollars, the bar the decision is made
      on) is under this may not be held; a day with unknown volume is kept and counted;
    * `delist_days` -- no position in the last `delist_days` days before a coin's last price,
      for a coin that stopped trading before the panel's end;
    * `only_listed` -- only coin-days the other venue lists may be held (a bool frame of the
      book's columns, indexed like the panel);
    * `max_instrument_share` -- each instrument's absolute weight is capped at this share of
      the book's gross exposure of that bar (the excess is cash, not moved to another coin);
    * `max_leg_contribution` -- one instrument's earning (price and funding) on one bar is
      clipped to this share of the capital in either direction. A stress on the result, not a
      rule a strategy could follow: the clipped money is simply not counted."""

    min_volume_usd: float | None = None
    delist_days: float | None = None
    max_instrument_share: float | None = None
    only_listed: pd.DataFrame | None = None
    max_leg_contribution: float | None = None

    def describe(self) -> str:
        if self.min_volume_usd is not None:
            return (f"coin-days with traded volume under ${self.min_volume_usd:,.0f} may not "
                    "be held")
        if self.delist_days is not None:
            return (f"no position in the last {self.delist_days:g} days before a coin's last "
                    "price")
        if self.max_instrument_share is not None:
            return (f"each instrument capped at {self.max_instrument_share:.0%} of the book's "
                    "gross exposure")
        if self.only_listed is not None:
            return "only coins the other venue lists may be held"
        if self.max_leg_contribution is not None:
            return (f"one instrument's earning on one bar clipped to "
                    f"{self.max_leg_contribution:.1%} of the capital in either direction")
        return "no change"

    @property
    def blocks_coin_days(self) -> bool:
        """Whether the rule is a ban on holding some coin-days (so the strategy can be asked to
        choose again without them, `reselect`)."""
        return (self.min_volume_usd is not None or self.delist_days is not None
                or self.only_listed is not None)


@dataclass(frozen=True)
class RuleEffect:
    weights: pd.DataFrame
    removed_bars: int
    """How many instrument-bars with a position the rule took away."""
    unknown_bars: int = 0
    """Instrument-bars the rule could not judge (volume unknown) and so kept."""
    first_row: int = 0
    """First row of the panel the counterfactual is valid from (a venue-listing rule is only
    defined from the other venue's first day)."""
    blocked: pd.DataFrame | None = None
    """For a ban on coin-days: the bool frame of banned coin-days over the whole panel."""


def blocked_coin_days(run: StoredRun, rule: Rule) -> tuple[pd.DataFrame, int, int] | None:
    """(banned coin-days as a bool frame like the prices, held coin-days with an unknown
    figure, first row the rule is defined from) for a ban rule; None for the other rules."""
    panel, held = run.panel, run.weights.abs() > ZERO_WEIGHT_TOL
    index = panel.prices.index
    if rule.min_volume_usd is not None:
        usd = panel.volume * panel.prices
        return usd < rule.min_volume_usd, int((usd.isna() & held).to_numpy().sum()), 0
    if rule.delist_days is not None:
        last = panel.prices.apply(lambda c: c.last_valid_index() if c.notna().any() else pd.NaT)
        stopped = last[last < index[-1] - pd.Timedelta(days=1)].dropna()
        window = pd.Timedelta(days=rule.delist_days)
        near = pd.DataFrame(False, index=index, columns=panel.prices.columns)
        for name, when in stopped.items():
            near.loc[(index >= when - window) & (index <= when), name] = True
        return near, 0, 0
    if rule.only_listed is not None:
        listed = rule.only_listed.reindex(index=index, columns=panel.prices.columns)
        listed = listed.fillna(False).astype(bool)
        first = int(np.argmax(listed.any(axis=1).to_numpy())) if listed.any(axis=None) else 0
        return ~listed, 0, first
    return None


def apply_rule(run: StoredRun, rule: Rule) -> RuleEffect:
    """The stored book with `rule` applied, the strategy NOT asked again (what was held and is
    banned simply becomes cash). Pure; the stored book is not modified."""
    weights = run.weights.copy()
    banned = blocked_coin_days(run, rule)
    if banned is not None:
        mask, unknown, first = banned
        gone = mask & (weights.abs() > ZERO_WEIGHT_TOL)
        return RuleEffect(weights.mask(gone, 0.0), int(gone.to_numpy().sum()), unknown,
                          first_row=first, blocked=mask)
    if rule.max_instrument_share is not None:
        gross = weights.abs().sum(axis=1)
        limit = (gross * rule.max_instrument_share).to_numpy()[:, None]
        values = weights.to_numpy()
        clipped = np.clip(values, -limit, limit)
        removed = int((np.abs(clipped - values) > ZERO_WEIGHT_TOL).sum())
        return RuleEffect(pd.DataFrame(clipped, index=weights.index, columns=weights.columns),
                          removed)
    return RuleEffect(weights, 0)


def reselect(run: StoredRun, blocked: pd.DataFrame) -> StoredRun:
    """The run with the banned coin-days made untradeable and the strategy ASKED AGAIN: it
    chooses among the coins that remain, as it would have if the venue had only those. Slower
    than `apply_rule` (the strategy runs once more) and closer to the question "what if the
    book could only trade these".

    The panel gets a snapshot id of its own: `qlab.strategies.retune` caches a book by snapshot
    id, columns and parameters, and returning the unmasked book for a masked panel is the trap
    this id (and the mask in the cache key, fixed 2026-10-28) closes."""
    import hashlib

    from qlab.harness.strategy import own_costs
    from qlab.pipeline.evaluate import resolve_strategy

    mask = blocked.reindex(index=run.panel.prices.index, columns=run.panel.prices.columns)
    mask = mask.fillna(False).astype(bool)
    tag = hashlib.sha256(mask.to_numpy().tobytes()).hexdigest()[:12]
    panel = dataclasses.replace(run.panel, snapshot_id=f"{run.panel.snapshot_id}:what-if-{tag}",
                                tradeable=run.panel.tradeable & ~mask)
    strategy = resolve_strategy(run.spec.code_ref)
    weights = strategy.target_weights(panel, run.spec.params)
    return dataclasses.replace(run, panel=panel, weights=weights,
                               extra_cost=own_costs(strategy, panel, run.spec.params))


def capped_leg_result(run: StoredRun, limit: float) -> RunResult:
    """The stored run with one instrument's earning (price and funding) on one bar clipped to
    +-`limit` of the capital; costs stay. See `Rule.max_leg_contribution`."""
    dec = decompose(run)
    earned = (dec.price + dec.funding).clip(-limit, limit)
    book = (earned - dec.cost).sum(axis=1) - dec.extra_cost
    base = run.backtest()
    capital_base = None if run.capital is None else run.capital.base
    if capital_base is None:
        capital = book.rename("capital_return")
    else:  # the clipped book on the strategy's own money, as `run_backtest` makes it
        capital = (book * capital_base.reindex(book.index).ffill().bfill().fillna(1.0)
                   ).rename("capital_return")
    return dataclasses.replace(base, net_return=book.rename("net_return"),
                               gross_return=earned.sum(axis=1).rename("gross_return"),
                               capital_return=capital)


__all__ = ["RECONCILE_TOLERANCE", "Decomposition", "Rule", "RuleEffect", "RunUnavailable",
           "StoredRun", "apply_rule", "blocked_coin_days", "capped_leg_result", "daily",
           "decompose", "load_run", "reselect"]
