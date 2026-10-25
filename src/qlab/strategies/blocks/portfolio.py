"""Portfolio blocks: turn scores into raw weights.

Each block gets the score table and a boolean table of eligible cells, and
returns weights that are zero wherever a coin is not eligible. Ranking is by
score, highest first; ties go to the coin listed first in the panel (so on a
tie it is the first to be long and the last to be short).

- `long_short`: long the `top_n` highest scores, short the `top_n` lowest. Each
  side sums to 0.5 in size, so the book has gross 1 and zero net. With fewer
  than `2 * top_n` eligible coins, each side takes `floor(eligible / 2)` coins;
  with fewer than two eligible coins there is no position.
- `long_only`: long the `top_n` highest scores in equal parts, summing to 1
  (fewer coins if fewer are eligible).
- `time_series_sign`: each eligible coin goes long if its own score is positive
  and short if negative, in parts of 1 / (number of eligible coins), so gross
  is at most 1. A score of exactly zero gives no position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.strategies.blocks._schema import Block, positive_int


def _ranks(scores: pd.DataFrame, eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    masked = scores.where(eligible)
    rank = masked.rank(axis=1, method="first", ascending=False)  # 1 = best, NaN if ineligible
    count = masked.notna().sum(axis=1)
    return rank, count


def long_short(scores: pd.DataFrame, eligible: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Long the best `top_n`, short the worst `top_n`; sides +0.5 / -0.5."""
    rank, count = _ranks(scores, eligible)
    k = np.minimum(top_n, count // 2)
    side = (0.5 / k.where(k > 0)).fillna(0.0)
    longs = rank.le(k, axis=0)
    shorts = rank.gt(count - k, axis=0)
    return longs.mul(side, axis=0) - shorts.mul(side, axis=0)


def long_only(scores: pd.DataFrame, eligible: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Long the best `top_n` equally, summing to 1."""
    rank, count = _ranks(scores, eligible)
    k = np.minimum(top_n, count)
    part = (1.0 / k.where(k > 0)).fillna(0.0)
    return rank.le(k, axis=0).mul(part, axis=0)


def time_series_sign(scores: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Sign of each coin's own score, divided by the number of eligible coins."""
    masked = scores.where(eligible)
    count = masked.notna().sum(axis=1)
    part = (1.0 / count.where(count > 0)).fillna(0.0)
    return np.sign(masked).fillna(0.0).mul(part, axis=0)


PORTFOLIOS: dict[str, Block] = {
    "long_short": Block(long_short, {"top_n": positive_int}),
    "long_only": Block(long_only, {"top_n": positive_int}),
    "time_series_sign": Block(time_series_sign, {}),
}
