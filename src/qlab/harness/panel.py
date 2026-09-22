"""Structural stand-in for the shared `MarketPanel` data contract.

The REAL implementation is `qlab.data.panel.MarketPanel`, built by a parallel
agent (see `docs/PLAN.md`, milestone M2, "слой данных"). This module defines
a minimal frozen dataclass with the same fields and the same shape contract
so the harness (`qlab.harness`) can be built and tested without waiting on
that work. The harness is written against the FIELDS described below, not
against this particular class — when `qlab.data.panel.MarketPanel` lands, it
should satisfy the same shape contract and can be used as a drop-in
replacement anywhere this stand-in is used.

Shape contract:
  - `prices`, `funding`, `tradeable`, `volume` share one UTC `DatetimeIndex`
    (rows) and one instrument column index (columns) — exactly the same
    index and columns on all four.
  - `prices`: instrument price, one column per instrument.
  - `funding`: a fraction; the funding rate for the period ENDING at that
    row's timestamp. `NaN` means "unknown". It must never be zero-filled by
    the harness or by a strategy — an unknown funding rate is not a zero
    funding rate (see `qlab.harness.accrual`).
  - `tradeable`: bool; point-in-time listing status. `False` means "no
    exposure permitted in this instrument at this timestamp" (delisted, not
    yet listed, halted, etc).
  - `volume` (docs/TASKS.md, T27): base-asset traded volume, same "NaN
    means unknown, never zero" contract as `funding` — see
    `qlab.data.panel.MarketPanel`'s own docstring for the full rationale
    (in particular why USD volume is never stored, only ever recomputed as
    `volume * prices`). Optional here and defaults to an all-NaN frame so
    every caller/test built before T27 keeps working unchanged; a real
    `qlab.data.panel.MarketPanel` always supplies a real one.
  - `meta`: free-form metadata about the snapshot (source, fetch time, ...).
    The harness does not read specific keys out of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    """Point-in-time market data for one snapshot. See module docstring.

    This is a light structural stand-in for `qlab.data.panel.MarketPanel`,
    not the real thing — see module docstring.
    """

    snapshot_id: str
    prices: pd.DataFrame
    funding: pd.DataFrame
    tradeable: pd.DataFrame
    meta: Mapping[str, object]
    volume: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.volume is None:
            object.__setattr__(
                self,
                "volume",
                pd.DataFrame(
                    float("nan"), index=self.prices.index, columns=self.prices.columns
                ),
            )
        for name, frame in (
            ("funding", self.funding),
            ("tradeable", self.tradeable),
            ("volume", self.volume),
        ):
            if not frame.index.equals(self.prices.index):
                raise ValueError(f"{name} index must match prices index exactly")
            if not frame.columns.equals(self.prices.columns):
                raise ValueError(f"{name} columns must match prices columns exactly")
