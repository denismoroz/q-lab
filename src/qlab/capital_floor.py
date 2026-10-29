"""The least money a strategy works on, measured (owner, 2026-10-26: «на каждую стратегию
выводить минимум возможный - чтобы стратегия работала и будем отталкиваться от него.
минимум должен быть обоснованный»).

The strategy's own code is run on a ladder of sums. The live strategies carry their money
in their params (`capital_usd`, `budget_cap_usdc`) and do what frab does with it: an order
below the venue's minimum is not placed, a flat transfer fee is a larger share of a small
book. So the book at $100 is not the book at $10 000 scaled down -- it is what the code
actually holds at $100.

A sum WORKS when all three hold (the criterion the owner chose on 2026-10-26, «Как большая
книга»):

1. every position is placeable and placed: the smallest order that opens a leg at that sum is
   at least the venue's minimum order (`min_leg_notional`; a strategy whose code does not model
   the minimum, like the stand's FRAB, is held to it all the same), and no position the
   reference book holds is dropped. A strategy that declares the orders it sent
   (`qlab.harness.strategy.own_orders`, docs/MIN_CAPITAL.md) is measured by them -- the same
   quantity as the stand's `min_capital_usd`, so the two cannot disagree about it -- and not by
   the weights, which read a position a cut left behind as a leg to buy and a balance that
   outgrew its working money as a smaller position;
2. its daily return per dollar moves with the reference book's: correlation at least
   `MIN_CORRELATION`, the owner's threshold for a paper book tracking the stand
   (seed/preregistration/2026-09-21-paper-bv2-trend.yaml, «>= 0.80»; qlab.paper_rule);
3. it earns after every cost: net return over the window above zero.

The reference is `REFERENCE_USD`, the owner's choice in the same answer. The minimum is the
smallest step from which this step and EVERY larger one works: a lucky pass at $75 under a
failure at $100 is not a floor. A strategy whose params carry no money (a book of weights,
`qlab.strategies.blocks`) cannot show what it does at a smaller sum; for it only the
position floor (`qlab.harness.metrics.min_capital_usd`) is known, and the result says so.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.harness.costs import CostModel
from qlab.harness.metrics import ZERO_WEIGHT_TOL, min_capital_usd
from qlab.harness.run import run_backtest
from qlab.harness.strategy import own_costs, own_orders
from qlab.pipeline.spec import StrategySpec, load_spec

#: The book every smaller sum is compared with (owner, 2026-10-26).
REFERENCE_USD = 10_000.0
#: The owner's tracking threshold, the same number qlab.paper_rule applies to paper books.
MIN_CORRELATION = 0.80
#: The sums tried. A measuring grid, not a judgement: the floor is reported as the step it
#: lands on, so its resolution is the gap to the next step down.
LADDER_USD: tuple[float, ...] = (50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500,
                                 2000, 3000, 5000, 7500, REFERENCE_USD)
#: Param names that hold a strategy's money in the live code (frab's own names).
MONEY_KEYS = ("capital_usd", "budget_cap_usdc")
#: Where measured floors are kept, one file per spec; and the version of this method -- a
#: stored floor is reused only for the same spec text and the same method.
STORE = Path("data/capital_floors")
METHOD_VERSION = "2026-10-29.1"


@dataclass
class Step:
    """One sum of the ladder."""

    usd: float
    dropped_positions: float  # share of the reference book's held cells not held here
    below_minimum: float  # share of the held cells smaller than the venue's minimum order
    correlation: float | None  # daily returns per dollar against the reference book
    net_return: float  # over the whole window, per dollar
    works: bool
    why_not: list[str] = field(default_factory=list)


@dataclass
class Floor:
    """The measured minimum of one strategy, with every step behind it."""

    idea_id: str
    floor_usd: float | None
    position_floor_usd: float | None  # smallest position >= the venue's minimum order
    money_in_params: bool
    window: tuple[str, str]
    steps: list[Step]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Floor:
        return cls(**{**data, "window": tuple(data["window"]),
                      "steps": [Step(**step) for step in data["steps"]]})

    def reasons_below(self) -> dict[str, float]:
        """For each criterion, the smallest step from which it holds at this and every larger
        step: which constraint sets the floor."""
        out: dict[str, float] = {}
        for name, failing in (("positions", lambda s: s.below_minimum > 0 or
                               s.dropped_positions > 0),
                              ("tracking", lambda s: s.correlation is None or
                               s.correlation < MIN_CORRELATION),
                              ("earns", lambda s: s.net_return <= 0)):
            low = None
            for step in sorted(self.steps, key=lambda s: s.usd, reverse=True):
                if failing(step):
                    break
                low = step.usd
            if low is not None:
                out[name] = low
        return out


def with_money(params: Mapping, usd: float) -> tuple[dict, bool]:
    """`params` with every money key (`MONEY_KEYS`, at any depth) set to `usd`; and whether
    any was found."""
    found = False

    def walk(value):
        nonlocal found
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                if key in MONEY_KEYS and isinstance(item, int | float):
                    out[key] = float(usd)
                    found = True
                else:
                    out[key] = walk(item)
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(params), found


def daily(returns: pd.Series) -> pd.Series:
    """Per-bar returns compounded into UTC days."""
    return (1.0 + returns).groupby(returns.index.normalize()).prod() - 1.0


def judge(usd: float, weights: pd.DataFrame, net: pd.Series, ref_weights: pd.DataFrame,
          ref_daily: pd.Series, min_leg_notional: float,
          orders: pd.DataFrame | None = None) -> Step:
    """One step against the reference book (pure). `orders`: the shares of the sum the
    strategy's orders that opened a leg took, in the weights' shape, when it declares them
    (`PlacedOrders.sizes` of the run at this sum); the position test then reads them instead
    of the weights."""
    held_ref = np.abs(ref_weights.to_numpy(dtype=float)) > ZERO_WEIGHT_TOL
    held = np.abs(weights.reindex_like(ref_weights).fillna(0.0).to_numpy(dtype=float)) \
        > ZERO_WEIGHT_TOL
    dropped = float((held_ref & ~held).sum() / held_ref.sum()) if held_ref.any() else 0.0
    mine = daily(net)
    both = pd.concat([mine, ref_daily], axis=1, join="inner").dropna()
    corr = None
    if len(both) >= 2 and both.iloc[:, 0].std() > 0 and both.iloc[:, 1].std() > 0:
        corr = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
    total = float((1.0 + net).prod() - 1.0)
    if orders is None:
        shares, measured_on, what = weights, weights, "positions"
    else:
        shares, measured_on, what = orders, orders, "opening orders"
    sizes = np.abs(shares.reindex_like(ref_weights).fillna(0.0).to_numpy(dtype=float)) * usd
    placed = np.abs(sizes) > ZERO_WEIGHT_TOL * usd
    cells = held if orders is None else placed
    below = float((cells & (sizes < min_leg_notional)).sum() / cells.sum()) if cells.any() else 0.0
    why: list[str] = []
    if below > 0:
        why.append(f"{below:.2%} of its {what} are below the ${min_leg_notional:g} minimum "
                   f"order (all fit from ${min_capital_usd(measured_on, min_leg_notional):,.0f})")
    if dropped > 0:
        why.append(f"{dropped:.2%} of the reference book's positions not held")
    if corr is None or corr < MIN_CORRELATION:
        why.append("daily returns do not track the reference book"
                   + ("" if corr is None else f" (correlation {corr:.2f})"))
    if total <= 0:
        why.append(f"net return {total:+.2%} over the window")
    return Step(usd=float(usd), dropped_positions=dropped, below_minimum=below,
                correlation=corr, net_return=total, works=not why, why_not=why)


def floor_from(steps: list[Step]) -> float | None:
    """The smallest step from which it and every larger step works (pure)."""
    floor = None
    for step in sorted(steps, key=lambda s: s.usd, reverse=True):
        if not step.works:
            break
        floor = step.usd
    return floor


def measure(spec: StrategySpec, panel, *, ladder: tuple[float, ...] = LADDER_USD) -> Floor:
    """Run `spec`'s strategy on every sum of `ladder` on `panel` and find its floor."""
    from qlab.pipeline.evaluate import resolve_strategy

    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    first = 0
    if spec.judged_from is not None:
        first = int(panel.prices.index.searchsorted(pd.Timestamp(spec.judged_from, tz="UTC")))

    def run(usd: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
        params, _ = with_money(spec.params, usd)
        strategy = resolve_strategy(spec.code_ref)
        weights = strategy.target_weights(panel, params)
        result = run_backtest(panel, weights, costs, panel.funding,
                              extra_cost=own_costs(strategy, panel, params))
        placed = own_orders(strategy, panel, params)
        return (weights.iloc[first:], result.net_return.iloc[first:],
                None if placed is None else placed.sizes.iloc[first:])

    _, has_money = with_money(spec.params, REFERENCE_USD)
    ref_weights, ref_net, ref_orders = run(REFERENCE_USD)
    # The position floor is the stand's `min_capital_usd`: by the orders a book-driving strategy
    # sent, by the weights otherwise (docs/MIN_CAPITAL.md).
    basis = ref_weights if ref_orders is None or not (
        np.abs(ref_orders.to_numpy(dtype=float)) > ZERO_WEIGHT_TOL).any() else ref_orders
    position_floor = (min_capital_usd(basis, spec.min_leg_notional)
                      if (np.abs(basis.to_numpy(dtype=float)) > ZERO_WEIGHT_TOL).any()
                      else None)
    index = ref_net.index
    window = (index[0].date().isoformat(), index[-1].date().isoformat()) if len(index) else ("", "")
    if not has_money:
        return Floor(idea_id=spec.idea_id, floor_usd=position_floor,
                     position_floor_usd=position_floor, money_in_params=False, window=window,
                     steps=[], note="the strategy's params carry no money: what it does at a "
                     "smaller sum cannot be run, only the position floor is known")
    ref_daily = daily(ref_net)
    steps = []
    for usd in ladder:
        weights, net, placed = ((ref_weights, ref_net, ref_orders) if usd == REFERENCE_USD
                                else run(usd))
        steps.append(judge(usd, weights, net, ref_weights, ref_daily, spec.min_leg_notional,
                           orders=placed))
    floor = floor_from(steps)
    reference = next(s for s in steps if s.usd == REFERENCE_USD)
    note = ("" if floor is not None else
            "does not work even at the reference sum: " + "; ".join(reference.why_not))
    return Floor(idea_id=spec.idea_id, floor_usd=floor, position_floor_usd=position_floor,
                 money_in_params=True, window=window, steps=steps, note=note)


def measure_spec(session, spec: StrategySpec, *, ladder: tuple[float, ...] = LADDER_USD
                 ) -> Floor:
    """`measure` on the spec's own data (the cached snapshot, cut where history is honest)."""
    from qlab.pipeline.evaluate import cut_truncated_history, resolve_panel

    return measure(spec, cut_truncated_history(resolve_panel(session, spec)), ladder=ladder)


def cached(session, spec_path: Path | str, *, store: Path | None = None) -> Floor:
    """The spec's floor from the store when it was measured on the same spec text with the same
    method, else measured now and stored."""
    path = Path(spec_path)
    folder = store if store is not None else STORE
    key = hashlib.sha256((METHOD_VERSION + path.read_text(encoding="utf-8")).encode()).hexdigest()
    target = folder / f"{path.stem}.json"
    if target.is_file():
        try:
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved.get("key") == key:
                return Floor.from_dict(saved["floor"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
    floor = measure_spec(session, load_spec(path))
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"key": key, "spec": str(path), "method": METHOD_VERSION,
                                  "floor": floor.to_dict()}, indent=1), encoding="utf-8")
    return floor


__all__ = ["LADDER_USD", "MIN_CORRELATION", "REFERENCE_USD", "Floor", "Step", "cached",
           "floor_from", "judge", "measure", "measure_spec", "with_money"]
