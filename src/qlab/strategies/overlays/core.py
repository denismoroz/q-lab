"""The overlay mechanism: pieces that change WHEN and HOW MUCH any strategy holds
(docs/OVERLAYS.md).

Owner, 2026-10-28: «возможно имеет смысл подумать про точку входа по стратегии. если мы знаем
что стратегия убыточна в определенный режим рынка - не входить»; «как тебе механизм
постепенного входа в стратегию - не для bv2 а вообще. входим и чуть чуть наращиваем позицию
если стратегия видит что она прибыльная или уменьшает позицию если прибыли нет»; «это все идеи
для платформы. было бы здорово проверять разные варианты возможно их можно смиксовать». The
mechanism is general: today's strategies are only its first users.

`Overlaid` wraps any strategy (`inner_code_ref`, `inner_params`) and an ordered list
`overlays: [{name, params}]`. Each overlay has one contract: it receives the target weights as
they stand (the inner strategy's, or the previous overlay's) and a `Context` -- the panel and the
inner strategy's OWN realised returns, computed from its un-overlaid weights on the stand the way
`qlab.meta_label` does -- and returns new weights. It may read nothing after the bar it decides
for: the look-ahead guard (`qlab.harness.lookahead`) runs on the whole `Overlaid`, and a
parametrised test runs every registered overlay at every point of its grid through it.

Two levels:

- **book level** (`BookOverlay`): one number per bar scales or gates the whole book;
- **position level** (`PositionOverlay`): acts per instrument -- how a new position is entered
  and how it is left.

Overlays chain in the declared order -- that is the "mix". A new overlay is one new module with
tests, registered by name (`register`); nothing else changes.

**Timing of the returns an overlay may read.** The inner strategy's return at decision row t is
earned over (t, t+1] (`qlab.harness.run`), so at row t it is realised only for rows before t:
`Context.known_returns` is that return shifted by one row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from qlab.boundedcache import BoundedCache
from qlab.harness.panel import MarketPanel
from qlab.strategies._inner import InnerCapital
from qlab.strategies._periods import periods_for

BOOK = "book"
POSITION = "position"
_REQUIRED = ("inner_code_ref", "inner_params", "overlays")


@dataclass(frozen=True)
class Context:
    """What an overlay may know at each bar, and nothing else."""

    panel: MarketPanel
    own_returns: pd.Series | None
    """The inner strategy's un-overlaid net return per decision row (earned over (t, t+1]),
    None when no overlay in the chain asked for it."""

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.panel.prices.index

    def bars(self, days: float) -> int:
        """Panel rows spanning `days` calendar days, at least one."""
        return periods_for(self.index, pd.Timedelta(days=days))

    @property
    def bars_per_day(self) -> float:
        return float(pd.Timedelta(days=1) / self.index.to_series().diff().median())

    def known_returns(self) -> pd.Series:
        """The inner strategy's net return realised by each bar: row t holds the return of row
        t - 1 (earned over (t - 1, t]), so it needs nothing after bar t. Zero before the first."""
        if self.own_returns is None:
            raise ValueError("this chain did not ask for the strategy's own returns")
        return self.own_returns.reindex(self.index).shift(1).fillna(0.0)


class Overlay:
    """See the module docstring. A subclass sets `name`, `level` and the allowed `params`."""

    name: ClassVar[str]
    level: ClassVar[str]
    needs_returns: ClassVar[bool] = False
    """Whether `apply` reads `Context.own_returns` (then the wrapper needs `costs`)."""
    params: ClassVar[tuple[str, ...]] = ()
    """The parameter names the overlay accepts; any other key is refused (a typo is not a
    silent default)."""
    short: ClassVar[str]
    """A short word for variant ids."""

    def check(self, params: Mapping[str, object]) -> None:
        """Refuse parameters the overlay cannot run with."""
        unknown = set(params) - set(self.params)
        missing = set(self.params) - set(params)
        if unknown or missing:
            raise ValueError(f"overlay {self.name!r}: params {sorted(self.params)} are all "
                             f"required, none has a default (unknown {sorted(unknown)}, "
                             f"missing {sorted(missing)})")

    def warmup_days(self, params: Mapping[str, object]) -> float:
        """Calendar days of history the overlay needs before it acts at all; before them it
        leaves the weights as they are (no evidence is not a reason to change anything)."""
        return 0.0

    def apply(self, weights: pd.DataFrame, ctx: Context, params: Mapping[str, object]
              ) -> pd.DataFrame:
        raise NotImplementedError

    def grid(self) -> dict[str, dict[str, object]]:
        """The overlay's family: label -> params. Never one hand-picked value."""
        raise NotImplementedError

    def coarse_grid(self) -> dict[str, dict[str, object]]:
        """A thinner family used inside pairs, so that two grids multiplied stay affordable."""
        return self.grid()


class BookOverlay(Overlay):
    """One number per bar scales (or switches off) the whole book."""

    level = BOOK

    def scale(self, ctx: Context, params: Mapping[str, object]) -> pd.Series:
        """The factor for each bar, >= 0, from what is known at that bar."""
        raise NotImplementedError

    def apply(self, weights: pd.DataFrame, ctx: Context, params: Mapping[str, object]
              ) -> pd.DataFrame:
        factor = self.scale(ctx, params).reindex(weights.index)
        if factor.isna().any() or (factor < 0).any() or not np.isfinite(factor.to_numpy()).all():
            raise ValueError(f"overlay {self.name!r} gave a scale that is not a finite number "
                             ">= 0")
        return weights.mul(factor, axis=0)


class PositionOverlay(Overlay):
    """Acts per instrument: how a position is entered or left."""

    level = POSITION


_REGISTRY: dict[str, Overlay] = {}


def register(overlay: Overlay) -> Overlay:
    """Register an overlay under its name. A name is taken once."""
    if overlay.name in _REGISTRY:
        raise ValueError(f"overlay {overlay.name!r} is registered already")
    _REGISTRY[overlay.name] = overlay
    return overlay


def get(name: str) -> Overlay:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown overlay {name!r}; registered: {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return list(_REGISTRY)


def parse_chain(raw: object) -> list[tuple[Overlay, dict[str, object]]]:
    """`[{name, params}, ...]` as (overlay, params) pairs, each checked."""
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError("overlays must be a list of {name, params}")
    chain = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) - {"name", "params"} or "name" not in item:
            raise ValueError(f"an overlay is {{name, params}}, got {item!r}")
        overlay = get(str(item["name"]))
        params = dict(item.get("params") or {})
        overlay.check(params)
        chain.append((overlay, params))
    return chain


def chain_warmup_days(raw: object) -> float:
    """The longest warm-up of any overlay of the chain, in calendar days."""
    return max((o.warmup_days(p) for o, p in parse_chain(raw)), default=0.0)


# --------------------------------------------------------------------------
# The wrapper

_CACHE_SIZE = 8
_CACHE: BoundedCache[tuple[pd.DataFrame, pd.Series | None]] = BoundedCache(_CACHE_SIZE)


def _fingerprint(panel: MarketPanel) -> str:
    """A cheap digest of the panel's prices: the snapshot id alone is not enough for a panel
    built in a test (the same id, other numbers)."""
    values = pd.util.hash_pandas_object(panel.prices, index=False).to_numpy()
    return hashlib.sha1(values.tobytes()).hexdigest()[:16]


def _inner_run(panel: MarketPanel, params: Mapping[str, object], need_returns: bool
               ) -> tuple[pd.DataFrame, pd.Series | None]:
    """The inner strategy's weights and, when asked, its net returns on the stand. Cached: a
    causal choice between many overlay settings runs the inner strategy once."""
    index = panel.prices.index
    costs = params.get("costs")
    key = (panel.snapshot_id, _fingerprint(panel), len(index), str(index[-1]),
           hash(tuple(panel.prices.columns)),
           json.dumps([params["inner_code_ref"], params["inner_params"], costs],
                      sort_keys=True, default=str), need_returns)
    if (hit := _CACHE.get(key)) is not None:
        return hit

    from qlab.pipeline.evaluate import resolve_strategy

    inner = resolve_strategy(str(params["inner_code_ref"]))
    weights = inner.target_weights(panel, dict(params["inner_params"]))  # type: ignore[arg-type]
    returns = None
    if need_returns:
        if not isinstance(costs, Mapping):
            raise ValueError("Overlaid: an overlay reads the strategy's own results, so "
                             "`costs` {taker_fee_bps, slippage_bps} are required (the spec's)")
        from qlab.harness.costs import CostModel
        from qlab.harness.run import run_backtest
        from qlab.harness.strategy import own_costs

        extra = own_costs(inner, panel, dict(params["inner_params"]))  # type: ignore[arg-type]
        returns = run_backtest(
            panel, weights,
            CostModel(taker_fee_bps=float(costs["taker_fee_bps"]),
                      slippage_bps=float(costs["slippage_bps"])),
            panel.funding, extra_cost=extra).net_return
    return _CACHE.put(key, (weights, returns))


class Overlaid(InnerCapital):
    """A strategy and a chain of overlays (module docstring). Every parameter is required."""

    name = "overlaid"

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [k for k in _REQUIRED if k not in params]
        if missing:
            raise ValueError(f"Overlaid needs params {missing}; none has a default")
        chain = parse_chain(params["overlays"])
        need = any(o.needs_returns for o, _ in chain)
        base, returns = _inner_run(panel, params, need)
        ctx = Context(panel=panel, own_returns=returns)
        weights = base
        for overlay, step in chain:
            weights = overlay.apply(weights, ctx, step)
            # Whatever an overlay does, a book never holds what cannot be traded, and an
            # unknown weight is not a decision (qlab.harness.strategy.validate_weights).
            weights = weights.fillna(0.0).where(panel.tradeable, 0.0)
        gross_before = float(base.abs().sum(axis=1).mean())
        gross_after = float(weights.abs().sum(axis=1).mean())
        self._last = {"overlay_gross_ratio": gross_after / gross_before if gross_before else 1.0}
        return weights

    def diagnostics(self) -> dict[str, float]:
        """How much of the strategy's average exposure the chain left (1.0 = all of it)."""
        return dict(self._last)

    def extra_costs(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.Series | None:
        """The inner strategy's own non-rate costs, as it states them for its un-overlaid book
        (None when it has none); the overlaid book pays the same flat fees."""
        from qlab.harness.strategy import own_costs
        from qlab.pipeline.evaluate import resolve_strategy

        inner = resolve_strategy(str(params["inner_code_ref"]))
        return own_costs(inner, panel, dict(params["inner_params"]))  # type: ignore[arg-type]


__all__ = ["BOOK", "POSITION", "BookOverlay", "Context", "Overlaid", "Overlay", "PositionOverlay",
           "chain_warmup_days", "get", "names", "parse_chain", "register"]
