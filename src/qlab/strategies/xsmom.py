"""Cross-sectional momentum on crypto perps -- qlab.strategies.xsmom.XsmomCrossSectionalMomentum.

**This is the strategy that caused q-lab to exist.** The previous validation
process rated it profitable (Sharpe 1.80, `research/xsmom_live_recon
/AUDIT_2026_09.md` line 68), it was funded with real money, and it lost
-11.19% over 13 weeks live (`AUDIT_2026_09.md` line 15: "живая торговля
2026-06-15...09-12: -11.19% за 13 недель, 0.7-й перцентиль"). Execution was
clean while it lost -- `AUDIT_2026_09.md` line 27: "corr(live, model) =
+0.86" -- so the loss is a signal finding, not an execution one
(`AUDIT_2026_09.md` line 30: "Убыток целиком «модельный»: сигнал выбрал то,
что упало."). The audit's own honesty ladder for the number that justified
funding it (`AUDIT_2026_09.md` lines 60-68):

    заявлено при запуске (2023+ survivor window)         Sharpe 1.80
    - выбор окна (2021+, same survivor universe)         Sharpe 1.21
    - survivorship (point-in-time, +28 dead coins)        Sharpe 0.76
    - funding drag (momentum longs what already ran up,
      so it structurally PAYS funding, -4.65%/yr at 20%
      vol -> -0.23 Sharpe)                                Sharpe ~=0.53

This module transcribes the LIVE engine's math (`frab/strategy/xsmom/{params,
strategy}.py` and `evaluators/{signal,rebalance}.py`) as faithfully as the
stateless `target_weights(panel, params) -> weights` interface allows, so
q-lab's own harness can reproduce (or fail to reproduce) that ladder
independently instead of quoting it. **The acceptance condition for this
transcription is inverted from every other strategy in this package: the
rules engine is expected to REJECT it.** If it does not, that is a finding
about the framework, not about XSMOM (docs/TASKS.md, T21).

Mechanism (the "cross-sectional long/short, periodic rebalance" shape named
in `qlab.harness.strategy.Strategy`'s own docstring): each configured
instrument gets a momentum-ensemble score -- for each lookback in `days`,
the trailing return is cross-sectionally z-scored against every other
instrument that has a defined return that day, then the z-scores are
averaged across lookbacks (`evaluators/signal.py::compute_scores`, ported
almost line-for-line below -- see `_ensemble_scores`). On a rebalance row
(weekly, anchored to a specific weekday -- `XsmomParams.rebalance_days` /
`anchor_dow`), the highest-scoring instruments go long and the
lowest-scoring go short, equal notional per leg, equal total notional both
sides (dollar-neutral by construction -- `XsmomParams.compute_notional_per_
position`: "Budget is split equally across both sides"). Between rebalance
rows the book is held flat, not re-decided daily -- see `_rebalance_rows`.

Left OUT -- named here, not silently dropped (same discipline as
`qlab.strategies.trend`'s and `qlab.strategies.frab`'s docstrings):

  - **The entire capital/margin envelope.** `XsmomParams.budget_cap`,
    `margin_buffer_factor`, `leverage` (the exchange margin setting, NOT an
    economic-exposure multiplier -- `leverage=1` means "fully collateralised
    per position", not "2x the book"), and `sizing_breakdown`'s reserve
    (`max(20.0, 0.08 * budget_cap)`, `params.py:144`) are live wallet/margin
    bookkeeping. `target_weights` returns fractions of book notional
    (`qlab.harness.strategy.Strategy`'s own contract), so book capital and
    margin safety are a `qlab evaluate --capital` / spec concern, not this
    function's. What IS transcribed from `sizing_breakdown` is the one
    number with an economic effect on the shape of the book: the $12
    minimum-leg floor (`params.py:141`, `_MIN_LEG = 12.0`) is carried
    through as this strategy's `specs/*.yaml` `min_leg_notional`, exactly
    like every other strategy in this package.
  - **The live capital-deployment quirk where fewer coins score than the
    configured leg count.** `evaluators/rebalance.py:144-187`: `notional`
    per leg is sized from `k` (the NOMINAL leg count from
    `XsmomParams.compute_k`), but only `effective_k = min(k, available //
    2)` legs are actually opened when fewer than `2*k` instruments have a
    defined score. The live engine therefore under-deploys the book rather
    than resizing remaining legs bigger -- this IS transcribed below
    (`target_weights`'s rebalance loop sizes every leg at `0.5 / k`, using
    the nominal `k`, and only fills `effective_k` of them), because it changes gross
    exposure and is cheap to keep faithful; it is named here because it is
    easy to mistake for a bug rather than a transcribed quirk.
  - **The live state machine.** `XsmomStrategy`/`states/*.py`: each position
    is driven `NEW -> OPENED -> CLOSE -> CLOSED` one exchange call at a
    time, with retry/failure handling (`strategy.py`'s `_advance_one`,
    20-iteration safety cap, `mark_failed` on any exception) and an hourly
    margin watchdog (`protection/margin_watchdog.py`) that can force-close a
    position between scheduled rebalances. None of this has an analogue in
    a pure `(panel, params) -> weights` function; the harness's realised
    P&L already assumes fills happen at the decided weight, which is the
    thing the state machine and watchdog exist to make true live.
  - **Funding accrual bookkeeping.** `actions/funding_accrual.py` accrues
    funding hourly from live exchange settlements into the DB. Economically
    this is the same effect `qlab.harness.run.run_backtest`'s own
    `accrual` parameter already applies from `panel.funding` -- nothing is
    lost, it is just computed by a different, already-existing layer.
  - **Mid-holding-period delisting or data gaps.** Between two rebalance
    rows, this function carries the prior rebalance's weights forward
    unconditionally, then (like `qlab.strategies.trend`) zeroes any weight
    where `panel.tradeable` is `False` as a final safety gate so
    `validate_weights` never sees exposure on a halted/delisted/bad-quote
    bar. Live, a name going bad mid-week would instead be caught by the
    hourly margin watchdog or a DROP/FLIP reconcile, which can act before
    the next scheduled Thursday. The stateless equivalent here can only
    zero the single leg -- for the bars affected, the book is transiently
    NOT dollar-neutral (one side short a leg). Measured on the honest
    discovered-universe spec (`specs/xsmom.yaml`, 222 instruments,
    2025-01-01..2026-09-20): 140 permanent closes this way across 92
    instruments (thin/short-lived names such as HMSTR getting delisted
    mid-holding, exactly the kind of name `params.py:12`'s own comment says
    got dropped from the live universe) -- NOT rare on a wide, honest
    universe, however rare it is on the live engine's small curated one.
    A SECOND, distinct effect also observed: 73 of those events are a
    `tradeable` flag flipping back to `True` a bar or two later (a brief
    data-quality exclusion, not a real delisting -- see
    `qlab.data.snapshot`'s bad-price-bar guard), which makes the carried
    weight silently reappear at its old, un-re-decided value once the gate
    reopens, rather than staying closed until the next scheduled Thursday.
    Both effects inflate turnover (and therefore cost) above the nominal
    weekly schedule -- confirmed by inspecting the turnover series directly
    (docs/XSMOM_T21.md's verification section), not assumed.
  - **`is_rebalance_due`'s true statefulness.** Live, this is a pure
    function of wall-clock time re-evaluated on every hourly tick against a
    `last_rebalance_ms` persisted in the DB (`evaluators/rebalance.py:50-
    75`): "never rebalanced -> always due" fires on whatever hour the
    strategy happens to be turned on, not necessarily the anchor weekday;
    every subsequent rebalance is gated to the anchor weekday AND
    `elapsed_days >= rebalance_days`. `_rebalance_rows` below reproduces
    this exactly on a bar-indexed panel (first bar of the window always
    rebalances; every later bar rebalances only if it lands on
    `anchor_dow` AND at least `rebalance_days` (converted to bars via
    `qlab.strategies._periods.periods_for`) have elapsed since the last
    one) -- it is a faithful discretisation of the same recurrence, not an
    approximation of a different rule.

Instrument universe -- the one deliberate methodological choice this module
makes, not a value read off any config file: `XsmomParams.universe`
defaults to `DEFAULT_XSMOM_UNIVERSE`, a 32-coin list the live engine treats
as an operator-overridable candidate pool (`params.py:10`: "Operators
narrow/override it via the UI"), always small and fixed in production --
XSMOM never ran against "every coin on the exchange." `universe=None` here
means exactly that hypothetical: every instrument the panel actually has is
a candidate, with no fixed list at all. This is what makes the two `specs/
xsmom*.yaml` files an apples-to-apples survivorship experiment
(docs/TASKS.md T21): `specs/xsmom.yaml` passes `universe: null` (honest
discovered universe, matching `specs/trend.yaml`'s own convention of no
fixed coin list); `specs/xsmom-frozen-universe.yaml` passes the literal
32-coin `DEFAULT_XSMOM_UNIVERSE`, i.e. the actual list the live engine
shipped with, restricted to a snapshot frozen 2026-06-12 by criteria
evaluated in June 2026 and applied backward to a 2025-01-01 backtest start
-- survivorship selection by construction. `XsmomParams.compute_k`'s
`universe_len` argument is `len(universe)` when a fixed list is given
(matching production exactly) and `len(panel.prices.columns)` when
`universe` is `None` (there being no other candidate count to use for the
tercile rule in that case) -- see `_compute_k`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies._periods import periods_for


def _compute_k(universe_len: int, n_positions: int | None) -> int:
    """`XsmomParams.compute_k`, `params.py:75-88`, transcribed verbatim.

    Auto mode (`n_positions is None`): tercile, `k = max(1, universe_len //
    3)`. Manual mode: `k = n_positions // 2` (`n_positions` is documented as
    "the *total* even count of long+short legs"). Either way, clamped to
    `[1, max(1, universe_len // 2)]` so a single side never exceeds half the
    universe.
    """
    if n_positions is None:
        k = max(1, universe_len // 3)
    else:
        k = n_positions // 2
    max_k = max(1, universe_len // 2) if universe_len >= 2 else 1
    return min(max(k, 1), max_k)


def _ensemble_scores(prices: pd.DataFrame, lookback_periods: Sequence[int]) -> pd.DataFrame:
    """`evaluators/signal.py::compute_scores`, `signal.py:69-95`, ported to a
    full-panel vectorised form (the source computes one row via a dict of
    per-coin close lists; this computes every row at once from `prices`,
    which is mathematically identical since each leg's z-score is a
    row-local operation -- `_zscore_cs` below never looks across rows).

    Per lookback `lb`: `momentum = price / price.shift(lb) - 1`, then
    cross-sectionally z-scored per row (`signal.py:74-77`, mean/std with
    `ddof=0` across whatever instruments have a defined momentum that row --
    pandas' default `skipna=True` reproduces this without extra code). The
    ensemble score for a (row, instrument) cell is the mean of its z-scored
    legs, but ONLY if EVERY lookback leg is defined for that cell
    (`signal.py:88-93`, `all_present`) -- one missing leg (insufficient
    history) makes the whole ensemble score `NaN` for that cell, not a
    partial average over fewer legs.
    """

    def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std = df.std(axis=1, ddof=0)
        return df.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)

    legs = []
    for lb in lookback_periods:
        momentum = prices / prices.shift(lb) - 1.0
        legs.append(_zscore_cs(momentum))

    arr = np.stack([leg.to_numpy() for leg in legs], axis=0)  # (n_lb, T, C)
    all_present = ~np.isnan(arr).any(axis=0)  # (T, C)
    with warnings.catch_warnings():
        # `signal.py:90-91`: an all-NaN cell (no lookback leg defined yet,
        # e.g. before any instrument has enough history) is expected and
        # papered over by the `all_present` mask right below -- not a
        # computation error worth a warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_arr = np.nanmean(arr, axis=0)
    mean_arr = np.where(all_present, mean_arr, np.nan)

    return pd.DataFrame(mean_arr, index=prices.index, columns=prices.columns)


def _rebalance_rows(index: pd.DatetimeIndex, rebalance_periods: int, anchor_dow: int) -> np.ndarray:
    """Bar-indexed discretisation of `is_rebalance_due`, `evaluators/
    rebalance.py:50-75`. See this module's docstring, "`is_rebalance_due`'s
    true statefulness", for why this is a faithful translation and not an
    approximation: rule 1 ("never rebalanced -> always due") fires on bar 0
    unconditionally; every later bar `i` is a rebalance iff at least
    `rebalance_periods` bars have elapsed since the last rebalance bar AND
    `index[i].weekday() == anchor_dow` (Python's `weekday()`, Monday=0,
    matching `rebalance.py:74`'s own convention exactly).
    """
    n = len(index)
    is_rebalance = np.zeros(n, dtype=bool)
    weekdays = index.weekday.to_numpy()
    last_rebalance_pos: int | None = None
    for i in range(n):
        if last_rebalance_pos is None:
            due = True
        else:
            due = (i - last_rebalance_pos) >= rebalance_periods and weekdays[i] == anchor_dow
        if due:
            is_rebalance[i] = True
            last_rebalance_pos = i
    return is_rebalance


class XsmomCrossSectionalMomentum:
    """Cross-sectional momentum-ensemble long/short, weekly rebalance,
    dollar-neutral. See module docstring for the audit context, the
    survivorship experiment this strategy exists to run, and everything the
    stateless interface below cannot express.

    Required `params` keys (no defaults -- see module docstring for where
    each is documented):

        universe: Sequence[str] | None
            `XsmomParams.universe`, default `DEFAULT_XSMOM_UNIVERSE`
            (`params.py:13-18,30`) -- the fixed 32-coin candidate list, or
            `None` for "every instrument the panel has" (this module's own
            honest-universe convention -- see module docstring's
            "Instrument universe" section; production never actually ran
            with `None`).
        lookbacks_days: Sequence[int]
            `XsmomParams.lookbacks`, default `(14, 21, 30, 45, 60)`
            (`params.py:35`).
        rebalance_days: int
            `XsmomParams.rebalance_days`, default `7` (`params.py:36`).
        anchor_dow: int
            `XsmomParams.anchor_dow`, default `3` (Thursday, Python
            `weekday()` convention) (`params.py:37`).
        n_positions: int | None
            `XsmomParams.n_positions`, default `None` -> auto tercile sizing
            via `compute_k` (`params.py:31`, `params.py:75-88`). An explicit
            even integer switches to manual mode (`k = n_positions // 2`).
    """

    name = "xsmom"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        universe: Sequence[str] | None = params["universe"]
        lookbacks_days: Sequence[int] = params["lookbacks_days"]
        rebalance_days: int = params["rebalance_days"]
        anchor_dow: int = params["anchor_dow"]
        n_positions: int | None = params["n_positions"]

        prices = panel.prices
        index = prices.index
        all_columns = prices.columns

        # `evaluators/rebalance.py:129-130`: the live engine only ever fetches
        # closes for `self._params.universe`, so the cross-sectional z-score
        # is computed across exactly that candidate pool -- never the whole
        # market, even when `universe` here stands in for "the whole market"
        # (see module docstring). Instruments named in a fixed `universe`
        # list but absent from this panel are simply not candidates (there
        # is no data to score them with).
        if universe is None:
            candidate_columns = list(all_columns)
            universe_len = len(all_columns)
        else:
            candidate_columns = [c for c in universe if c in all_columns]
            universe_len = len(universe)

        k = _compute_k(universe_len, n_positions)

        lookback_periods = [periods_for(index, pd.Timedelta(days=d)) for d in lookbacks_days]
        candidate_prices = prices[candidate_columns]
        scores = _ensemble_scores(candidate_prices, lookback_periods)

        rebalance_periods = periods_for(index, pd.Timedelta(days=rebalance_days))
        rebalance_mask = _rebalance_rows(index, rebalance_periods, anchor_dow)

        raw = pd.DataFrame(0.0, index=index, columns=all_columns)
        per_leg_weight = 0.5 / k

        for pos in np.flatnonzero(rebalance_mask):
            row_label = index[pos]
            row_scores = scores.loc[row_label]
            row_tradeable = panel.tradeable.loc[row_label, candidate_columns]

            # Only a currently-tradeable, currently-scored instrument is a
            # real candidate -- `evaluators/rebalance.py:174-176` filters to
            # `scores.items()` (which is already history-gated by
            # `all_present` above) intersected with `universe`; the
            # tradeable filter is this module's own addition so the
            # long/short book built here is exactly dollar-neutral at the
            # moment it is decided (see module docstring's "Mid-holding-
            # period delisting" note for why this can still drift between
            # rebalances).
            eligible = row_scores[row_tradeable.reindex(row_scores.index).fillna(False)].dropna()
            available = len(eligible)
            effective_k = min(k, available // 2) if available >= 2 else 0
            if effective_k == 0:
                continue

            # `evaluators/rebalance.py:172-187`: sort descending by score,
            # top `effective_k` long, bottom `effective_k` short. Stable
            # sort to match Python's `sorted()` tie-break (original order),
            # same as the source.
            ranked = eligible.sort_values(ascending=False, kind="mergesort")
            long_names = ranked.index[:effective_k]
            short_names = ranked.index[-effective_k:]

            raw.loc[row_label, long_names] = per_leg_weight
            raw.loc[row_label, short_names] = -per_leg_weight

        # Hold flat between rebalances: only rebalance rows carry a decided
        # value above, everywhere else is 0.0 (never a real decision) until
        # forward-filled from the last rebalance -- module docstring,
        # "Mechanism".
        weights = raw.where(pd.Series(rebalance_mask, index=index), np.nan).ffill().fillna(0.0)

        # Final safety gate, same principle as `qlab.strategies.trend`:
        # never carry exposure into a bar where the instrument is not
        # tradeable, even if that exposure was decided at an earlier,
        # valid rebalance (module docstring, "Mid-holding-period delisting").
        weights = weights.where(panel.tradeable, 0.0)

        return weights
