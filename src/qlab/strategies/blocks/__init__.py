"""Composable strategy blocks: write a spec, not Python.

`Composed` (code_ref `qlab.strategies.blocks:Composed`) builds target weights
from four kinds of ready, tested blocks: a universe, a signal, a portfolio rule
and a sizing rule. Adding a block is one function plus one entry in the
registry dict of its module (`UNIVERSES`, `SIGNALS`, `PORTFOLIOS`, `SIZINGS`).
"""

from qlab.strategies.blocks.composed import Composed

__all__ = ["Composed"]
