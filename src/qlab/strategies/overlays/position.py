"""Position-level overlays: how a position is entered and how it is left (docs/OVERLAYS.md).

Owner, 2026-10-28: «как тебе механизм постепенного входа в стратегию ... входим и чуть чуть
наращиваем позицию если стратегия видит что она прибыльная или уменьшает позицию если прибыли
нет»; «возможно эта штука будет лучше работать в xsmom или trend - когда мы открываем или
закрываем позиции».

- `gradual_entry`: a new or increased position is taken in `tranches` parts, the first at once and
  the last `over_days` later. Variant `only_if_profit`: the next part is added only while the
  position is in profit since it was entered (adding to winners, as the classic trend followers
  did). The size of each part is the remaining gap to the CURRENT target divided by the parts
  left, so a target that moves meanwhile is followed and the last part always reaches it.
- `gradual_exit`: a position that has to shrink or close does so in `tranches` parts over
  `over_days` the same way.

Details that are decisions, not accidents:

- a position that must change SIDE is closed at once and the new one is entered gradually;
- an instrument that cannot be traded at a bar is flat at it (`MarketPanel.tradeable`), whatever
  was held;
- the profit test uses the volume-weighted average price of the parts already taken and the price
  of the current bar -- both known at the bar;
- while a position is still being built, the book is smaller than the strategy's own; while one is
  being left, it is larger. The overlay does not net these against other instruments.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.strategies.overlays.core import Context, PositionOverlay, register


def _spacing(over_bars: int, tranches: int) -> int:
    """Bars between two parts so that the last one falls `over_bars` after the first."""
    return max(1, round(over_bars / (tranches - 1))) if tranches > 1 else 1


def ladder(targets: pd.DataFrame, prices: pd.DataFrame, tradeable: pd.DataFrame, *,
           tranches_in: int, gap_in: int, only_if_profit: bool,
           tranches_out: int, gap_out: int) -> pd.DataFrame:
    """The held weights when each instrument walks to its target in parts (see the module
    docstring). With `tranches_in` / `tranches_out` of 1 that direction is immediate."""
    target = targets.to_numpy(dtype=float)
    price = prices.reindex_like(targets).to_numpy(dtype=float)
    can_trade = tradeable.reindex_like(targets).to_numpy(dtype=bool)
    rows, cols = target.shape
    held = np.zeros((rows, cols))
    prev = np.zeros(cols)
    avg = np.full(cols, np.nan)
    step_in = np.zeros(cols, dtype=int)  # parts already taken of the entry in progress
    step_out = np.zeros(cols, dtype=int)
    last_in = np.full(cols, -(10**9))
    last_out = np.full(cols, -(10**9))
    for t in range(rows):
        tgt = target[t]
        usable = can_trade[t]
        if not usable.all():  # a position cannot be held where nothing can be traded
            gone = ~usable
            prev = np.where(gone, 0.0, prev)
            avg = np.where(gone, np.nan, avg)
            step_in[gone] = 0
            step_out[gone] = 0
        if not (step_in.any() or step_out.any()) and np.array_equal(tgt, prev):
            held[t] = prev
            continue
        px = price[t]
        live = usable & np.isfinite(px)
        flip = prev * tgt < 0
        cur = np.where(flip, 0.0, prev)  # a position that changes side is closed at once
        avg = np.where(flip, np.nan, avg)
        step_in[flip] = 0
        step_out[flip] = 0
        grow = (np.abs(tgt) > np.abs(cur)) & live
        shrink = (np.abs(tgt) < np.abs(cur))
        new = cur.copy()

        # Entry: the first part at once, the next ones every `gap_in` bars.
        started = grow & (step_in == 0)
        in_profit = np.ones(cols, dtype=bool)
        if only_if_profit:
            with np.errstate(invalid="ignore", divide="ignore"):
                in_profit = np.sign(cur) * (px / avg - 1.0) > 0
        due = (grow & (step_in > 0) & (step_in < tranches_in)
               & (t - last_in >= gap_in) & in_profit)
        take = started | due
        left = np.where(started, tranches_in, tranches_in - step_in)
        add = np.where(take, (tgt - cur) / np.maximum(left, 1), 0.0)
        new = np.where(take, cur + add, new)
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted = np.where(np.abs(cur) > 0, avg, 0.0) * np.abs(cur) + px * np.abs(add)
            avg = np.where(take, weighted / np.abs(new), avg)
        step_in = np.where(take, step_in + 1, step_in)
        last_in = np.where(take, t, last_in)
        finished = take & (step_in >= tranches_in)
        step_in = np.where(finished | ~grow, 0, step_in)

        # Exit: the first part at once, the next ones every `gap_out` bars.
        began = shrink & (step_out == 0)
        due_out = shrink & (step_out > 0) & (step_out < tranches_out) & (t - last_out >= gap_out)
        cut = began | due_out
        left_out = np.where(began, tranches_out, tranches_out - step_out)
        new = np.where(cut, cur + (tgt - cur) / np.maximum(left_out, 1), new)
        step_out = np.where(cut, step_out + 1, step_out)
        last_out = np.where(cut, t, last_out)
        done_out = cut & (step_out >= tranches_out)
        step_out = np.where(done_out | ~shrink, 0, step_out)

        # A target met exactly is met exactly (no float dust from the divisions).
        new = np.where(np.isclose(new, tgt, rtol=0.0, atol=1e-15), tgt, new)
        avg = np.where(new == 0.0, np.nan, avg)
        held[t] = new
        prev = new
    return pd.DataFrame(held, index=targets.index, columns=targets.columns)


class GradualEntry(PositionOverlay):
    name = "gradual_entry"
    short = "enter"
    params = ("tranches", "over_days", "only_if_profit")

    def check(self, params: Mapping[str, object]) -> None:
        super().check(params)
        if int(params["tranches"]) < 2:  # type: ignore[call-overload]
            raise ValueError("gradual_entry: tranches is at least 2 (1 would be no overlay)")
        if float(params["over_days"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("gradual_entry: over_days must be > 0")
        if not isinstance(params["only_if_profit"], bool):
            raise ValueError("gradual_entry: only_if_profit is true or false")

    def apply(self, weights: pd.DataFrame, ctx: Context, params: Mapping[str, object]
              ) -> pd.DataFrame:
        k = int(params["tranches"])  # type: ignore[call-overload]
        gap = _spacing(ctx.bars(float(params["over_days"])), k)  # type: ignore[arg-type]
        return ladder(weights, ctx.panel.prices, ctx.panel.tradeable,
                      tranches_in=k, gap_in=gap, only_if_profit=bool(params["only_if_profit"]),
                      tranches_out=1, gap_out=1)

    def grid(self) -> dict[str, dict[str, object]]:
        return {f"k{k}-d{d}-{'profit' if p else 'plain'}":
                {"tranches": k, "over_days": d, "only_if_profit": p}
                for k, d, p in itertools.product((2, 3, 5), (3, 7, 14), (False, True))}

    def coarse_grid(self) -> dict[str, dict[str, object]]:
        return {f"k{k}-d{d}-{'profit' if p else 'plain'}":
                {"tranches": k, "over_days": d, "only_if_profit": p}
                for k, d, p in itertools.product((2, 3), (3, 7), (False, True))}


class GradualExit(PositionOverlay):
    name = "gradual_exit"
    short = "exit"
    params = ("tranches", "over_days")

    def check(self, params: Mapping[str, object]) -> None:
        super().check(params)
        if int(params["tranches"]) < 2:  # type: ignore[call-overload]
            raise ValueError("gradual_exit: tranches is at least 2 (1 would be no overlay)")
        if float(params["over_days"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("gradual_exit: over_days must be > 0")

    def apply(self, weights: pd.DataFrame, ctx: Context, params: Mapping[str, object]
              ) -> pd.DataFrame:
        k = int(params["tranches"])  # type: ignore[call-overload]
        gap = _spacing(ctx.bars(float(params["over_days"])), k)  # type: ignore[arg-type]
        return ladder(weights, ctx.panel.prices, ctx.panel.tradeable,
                      tranches_in=1, gap_in=1, only_if_profit=False,
                      tranches_out=k, gap_out=gap)

    def grid(self) -> dict[str, dict[str, object]]:
        return {f"k{k}-d{d}": {"tranches": k, "over_days": d}
                for k, d in itertools.product((2, 3), (3, 7))}


GRADUAL_ENTRY = register(GradualEntry())
GRADUAL_EXIT = register(GradualExit())

__all__ = ["GRADUAL_ENTRY", "GRADUAL_EXIT", "GradualEntry", "GradualExit", "ladder"]
