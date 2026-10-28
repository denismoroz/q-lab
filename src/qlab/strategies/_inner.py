"""A wrapper strategy's money is its inner strategy's (docs/CAPITAL_BASE.md).

`RegimeGate`, `SplitCadence`, `Overlaid` and the calibration's `NoiseStrategy` change WHEN or
HOW MUCH an inner strategy holds, not what the money is: the weights they return are still
shares of the inner strategy's book. So the base of those weights (`own_base`) and whether the
profit is put back into positions (`profit_reinvested`) are the inner strategy's answers.

For a wrapper that changes how much is held (a gate that keeps the book in cash, a volatility
target) the inner book's own equity path is the one imagined to run underneath; that is an
approximation, stated in docs/CAPITAL_BASE.md, and exact whenever the inner strategy has no
base of its own (trend, XSMOM, FRAB: base 1).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import own_base, profit_reinvested


class InnerCapital:
    """Mixin for a wrapper whose params name the inner strategy: `inner_code_key` holds its
    `code_ref` and `inner_params_key` its params (the defaults are the wrappers' own)."""

    inner_code_key = "inner_code_ref"
    inner_params_key = "inner_params"

    def _inner(self, params: Mapping[str, object]) -> tuple[object, dict[str, object]]:
        from qlab.pipeline.evaluate import resolve_strategy

        return (resolve_strategy(str(params[self.inner_code_key])),
                dict(params[self.inner_params_key]))  # type: ignore[call-overload]

    def own_base(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.Series | None:
        inner, inner_params = self._inner(params)
        return own_base(inner, panel, inner_params)

    def profit_reinvested(self, params: Mapping[str, object]) -> bool:
        inner, inner_params = self._inner(params)
        return profit_reinvested(inner, inner_params)
