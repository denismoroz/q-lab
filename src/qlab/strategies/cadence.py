"""Different timeframes for entry and exit (owner, 2026-10-01; docs/TASKS.md T25):
"возможно нужно использовать разные тайфреймы для входа и разные для выхода".

`SplitCadence` wraps any strategy and changes only WHEN its decisions may be
acted on. The wrapped strategy runs on the panel's own (fast) bars and says
what it wants at every bar; the wrapper lets the book

- **grow** (open a position, add to it, flip it) only on bars that close an
  entry period -- e.g. once a day, at the daily close;
- **shrink** (reduce or close it) on bars that close an exit period -- e.g.
  every hour.

No exit rule is invented here: what to exit, and when it is wanted, is the
wrapped strategy's own signal. The wrapper only decides how often that wish is
checked. With both periods equal to the panel's bar it is the wrapped strategy
unchanged (tested).

A bar "closes" a period P when the bar's END (its timestamp plus one bar) is
a multiple of P counted from the Unix epoch in UTC: on 1h bars the 23:00 bar
closes the day. Weights decided at that bar earn the next interval, as
everywhere in the harness (`qlab.harness.strategy`, alignment rule).

Validity on a timeframe is the wrapped strategy's: the wrapper refuses to run
if the wrapped strategy declares `valid_intervals` that exclude the panel's
bar size (T25) -- a strategy wrong on 1h bars stays wrong behind a wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.data.sources.base import INTERVAL_TO_TIMEDELTA
from qlab.harness.panel import MarketPanel

_REQUIRED = ("inner_code_ref", "inner_params", "entry_every", "exit_every")


def closes_period(index: pd.DatetimeIndex, period: pd.Timedelta) -> np.ndarray:
    """Boolean per row: does this bar end exactly on a multiple of `period`?"""
    bar = index.to_series().diff().median()
    # Timedelta arithmetic, not raw integers: pandas 3 stores this index in
    # microseconds while a Timedelta's `.value` is nanoseconds.
    since_epoch = (index + bar) - pd.Timestamp(0, tz="UTC")
    return np.asarray((since_epoch % period) == pd.Timedelta(0))


class SplitCadence:
    """See module docstring. All params are required."""

    name = "split-cadence"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [key for key in _REQUIRED if key not in params]
        if missing:
            raise ValueError(f"SplitCadence needs params {missing}; none has a default")

        # Imported here, not at module level: qlab.pipeline imports the
        # strategies package's modules by code_ref, and this keeps the
        # dependency one-way at import time.
        from qlab.pipeline.evaluate import resolve_strategy

        inner = resolve_strategy(str(params["inner_code_ref"]))
        index = panel.prices.index
        bar = index.to_series().diff().median()

        valid = getattr(inner, "valid_intervals", None)
        if valid is not None and not any(INTERVAL_TO_TIMEDELTA[i] == bar for i in valid):
            raise ValueError(
                f"{params['inner_code_ref']} is not valid on {bar} bars "
                f"(declares valid_intervals={tuple(valid)})"
            )

        entry_every = pd.Timedelta(str(params["entry_every"]))
        exit_every = pd.Timedelta(str(params["exit_every"]))
        for label, period in (("entry_every", entry_every), ("exit_every", exit_every)):
            if period < bar or period % bar != pd.Timedelta(0):
                raise ValueError(f"{label}={period} must be a whole number of {bar} bars")

        wanted = inner.target_weights(panel, dict(params["inner_params"]))  # type: ignore[arg-type]
        want = wanted.to_numpy(dtype=float)
        tradeable = panel.tradeable.to_numpy(dtype=bool)
        entry_rows = closes_period(index, entry_every)
        exit_rows = closes_period(index, exit_every)

        held = np.zeros(want.shape[1])
        out = np.zeros_like(want)
        for t in range(len(index)):
            target = want[t]
            if entry_rows[t]:
                held = target.copy()
            elif exit_rows[t]:
                same_side = np.sign(target) == np.sign(held)
                smaller = np.abs(target) <= np.abs(held)
                shrink = same_side & smaller
                held = np.where(shrink, target, np.where(same_side, held, 0.0))
            held = np.where(tradeable[t], held, 0.0)
            out[t] = held
        return pd.DataFrame(out, index=index, columns=panel.prices.columns)


__all__ = ["SplitCadence", "closes_period"]
