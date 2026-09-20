"""``MarketPanel`` — the one contract every other part of the framework
consumes (docs/PLAN.md, M2 "слой данных").

A panel bundles three aligned frames (prices, funding, tradeable) plus
provenance metadata, anchored to a ``data_snapshot`` row by ``snapshot_id``.
Nothing downstream (harness, strategies) is allowed to read raw exchange
data directly — everything goes through this shape, so a strategy backtest
can never accidentally see a different index/column layout for prices than
for funding or tradeability.

Point-in-time correctness is the whole point of ``tradeable``: an instrument
that delisted mid-range must read ``True`` before its delisting and ``False``
after, and one that only listed mid-range must read ``False`` before it
listed. A panel that is all-``True`` for a range where the venue actually
delisted something is survivorship-biased — this is the exact defect
documented in funding-rate-arbitrage's XSMOM stress test
(research/cross_sectional/crypto/survivorship.py), which measured it costing
the book 0.45 of Sharpe and ~20 points of annual return once dead coins were
included point-in-time instead of silently dropped. ``build_snapshot``
(qlab.data.snapshot) is responsible for actually computing ``tradeable``
that way; this module only enforces the shape once it exists.

``prices`` may contain NaN where an instrument did not trade in a bar.
``funding`` NaN means the funding rate for that period is *unknown* — never
zero. A harness must never treat missing funding as free-to-hold; it should
refuse to run over any window where a tradeable instrument has NaN funding
rather than silently substituting zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import pandas as pd

_FRAME_NAMES = ("prices", "funding", "tradeable")


def _require_utc_datetime_index(index: pd.Index, label: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{label}.index must be a pandas DatetimeIndex, got {type(index)!r}")
    if index.tz is None:
        raise ValueError(f"{label}.index must be timezone-aware (UTC), got a naive index")
    if str(index.tz) != "UTC":
        raise ValueError(f"{label}.index must be UTC, got tz={index.tz!r}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label}.index must be sorted ascending")
    if not index.is_unique:
        raise ValueError(f"{label}.index must not contain duplicate timestamps")


@dataclass(frozen=True)
class MarketPanel:
    """Point-in-time market data for a fixed set of instruments.

    Attributes
    ----------
    snapshot_id:
        sha256 id of the underlying ``data_snapshot`` registry row
        (qlab.registry.models.DataSnapshot). Identifies *where this data
        came from*, not the current view — ``slice``/``restrict``/``align``
        return new panels that keep the same ``snapshot_id`` because they
        are views over the same recorded fetch, not new snapshots.
    prices:
        index=UTC ``DatetimeIndex``, columns=instrument. Mark/close price.
        NaN means the instrument did not trade in that bar.
    funding:
        Same shape as ``prices``. The funding rate (as a fraction, e.g.
        0.0001 for 1bp) for the period *ending* at that index label. NaN
        means unknown — it must never be read as zero.
    tradeable:
        Same shape, dtype bool. Whether the instrument was listed and
        tradeable at that time — this is what makes the panel point-in-time
        rather than survivor-biased.
    meta:
        Free-form provenance: at minimum ``venue``, ``interval``,
        ``fetched_at``, the requested date range, and
        ``universe_complete`` (bool).

        ``universe_complete`` closes the other half of the survivorship gap
        that ``tradeable`` alone cannot: ``tradeable`` is honest about the
        instruments that ARE in the panel, but says nothing about coins that
        were never requested at all. A panel built from a hand-typed
        instrument list (the tickers a person remembers — i.e. the
        survivors) is just as survivorship-biased as an all-``True``
        ``tradeable`` frame, even though every column in it is individually
        point-in-time correct. ``universe_complete=True`` means the
        instrument list came from a source's ``discover_universe()``
        (survivors + delisted); ``False`` means it was supplied by the
        caller, or the source's free API cannot expose delisted names at all
        (e.g. Binance) so a complete universe was never possible. The
        registry rule ``honest_universe`` (metric ``point_in_time_universe``)
        reads this flag to fail-closed on hand-picked panels.
    """

    snapshot_id: str
    prices: pd.DataFrame
    funding: pd.DataFrame
    tradeable: pd.DataFrame
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in _FRAME_NAMES:
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"{name} must be a pandas DataFrame, got {type(frame)!r}")

        for name in _FRAME_NAMES:
            _require_utc_datetime_index(getattr(self, name).index, name)

        if not self.prices.index.equals(self.funding.index):
            raise ValueError("prices.index and funding.index must be identical")
        if not self.prices.index.equals(self.tradeable.index):
            raise ValueError("prices.index and tradeable.index must be identical")

        price_cols = list(self.prices.columns)
        if list(self.funding.columns) != price_cols:
            raise ValueError("prices.columns and funding.columns must be identical (same order)")
        if list(self.tradeable.columns) != price_cols:
            raise ValueError("prices.columns and tradeable.columns must be identical (same order)")

        non_bool = [
            c for c, dt in self.tradeable.dtypes.items() if not pd.api.types.is_bool_dtype(dt)
        ]
        if non_bool:
            raise ValueError(f"tradeable must be all-bool columns, got non-bool: {non_bool}")

        universe_complete = self.meta.get("universe_complete")
        if universe_complete is not None and not isinstance(universe_complete, bool):
            raise ValueError("meta['universe_complete'] must be a bool when present")

    @property
    def instruments(self) -> list[str]:
        return list(self.prices.columns)

    def slice(self, start, end) -> MarketPanel:
        """Return a panel restricted to ``[start, end]`` (inclusive) of the
        index. ``snapshot_id``/``meta`` (including ``universe_complete``)
        are unchanged — narrowing the time range doesn't drop or hand-pick
        any instrument, so completeness of the universe is unaffected."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        if start_ts > end_ts:
            raise ValueError(f"slice start {start_ts} is after end {end_ts}")

        return replace(
            self,
            prices=self.prices.loc[start_ts:end_ts],
            funding=self.funding.loc[start_ts:end_ts],
            tradeable=self.tradeable.loc[start_ts:end_ts],
        )

    def restrict(self, instruments: Sequence[str]) -> MarketPanel:
        """Return a panel restricted to the given instruments, in the given
        order. Raises if any requested instrument isn't in this panel.

        Always sets ``meta["universe_complete"] = False`` on the result,
        even if ``instruments`` happens to be the full current column list:
        naming an explicit instrument list is a manual selection by
        construction, and the whole point of the flag is to catch exactly
        this — a caller picking the coins it remembers rather than the
        source's discovered universe. Only ``build_snapshot`` (via a
        source's ``discover_universe``) is allowed to claim ``True``.
        """
        requested = list(instruments)
        missing = [i for i in requested if i not in self.prices.columns]
        if missing:
            raise ValueError(f"instruments not in panel: {missing}")

        return replace(
            self,
            prices=self.prices[requested],
            funding=self.funding[requested],
            tradeable=self.tradeable[requested],
            meta=self._meta_with(universe_complete=False),
        )

    def align(self, other: MarketPanel) -> tuple[MarketPanel, MarketPanel]:
        """Restrict both panels to their common index and common
        instruments (intersection). Each returned panel keeps its own
        ``snapshot_id`` — aligning two panels combines views, it does not
        merge their provenance into a new snapshot. Both results get
        ``meta["universe_complete"] = False``: the intersection of two
        instrument sets is a derived, ad hoc subset, never a source's
        discovered universe, regardless of what either panel started with.
        """
        common_index = self.prices.index.intersection(other.prices.index)
        common_columns = sorted(set(self.prices.columns) & set(other.prices.columns))
        if len(common_index) == 0:
            raise ValueError("panels have no overlapping timestamps")
        if not common_columns:
            raise ValueError("panels have no overlapping instruments")

        return (
            self._select(common_index, common_columns),
            other._select(common_index, common_columns),
        )

    def _meta_with(self, **overrides: object) -> dict[str, object]:
        merged = dict(self.meta)
        merged.update(overrides)
        return merged

    def _select(self, index: pd.DatetimeIndex, columns: Iterable[str]) -> MarketPanel:
        columns = list(columns)
        return replace(
            self,
            prices=self.prices.loc[index, columns],
            funding=self.funding.loc[index, columns],
            tradeable=self.tradeable.loc[index, columns],
            meta=self._meta_with(universe_complete=False),
        )


__all__ = ["MarketPanel"]
