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
- `long_short_quantile`: the way the published papers sort coins. Split the eligible
  coins into `quantiles` equal groups (5 = fifths, 3 = thirds, 10 = tenths); each
  group has k = floor(eligible / quantiles) coins. Long the top group, short the
  bottom group; each side sums to 0.5 (gross 1, zero net). With k = 0 there is no
  position. `quantiles` is at least 2.
- `long_only_quantile`: long the top group of k = floor(eligible / quantiles)
  coins in equal parts, summing to 1; with k = 0 there is no position.
  `quantiles` is at least 1 (1 = every eligible coin).
- `long_short_fraction`: the same as `long_short_quantile` but the group is a share of the
  eligible coins, as the papers that use "the top 30%" do: k = floor(eligible * `fraction`)
  coins per side, `fraction` in (0, 0.5]. Long the top k, short the bottom k; sides +0.5 / -0.5.
  With k = 0 there is no position. (The product is nudged by 1e-9 before flooring, so 100 coins
  at 0.29 give 29 and not 28 through float noise.)
- `long_only_fraction`: long the top k = floor(eligible * `fraction`) coins in equal parts,
  summing to 1; `fraction` in (0, 1]. With k = 0 there is no position.
- `spot_perp_carry`: funding harvest, a coin against its perpetual. See the function.
- `time_series_sign`: each eligible coin goes long if its own score is positive
  and short if negative, in parts of 1 / (number of eligible coins), so gross
  is at most 1. A score of exactly zero gives no position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.strategies.blocks._schema import Block, fraction_up_to, positive_int, quantile_count


def _ranks(scores: pd.DataFrame, eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    masked = scores.where(eligible)
    rank = masked.rank(axis=1, method="first", ascending=False)  # 1 = best, NaN if ineligible
    count = masked.notna().sum(axis=1)
    return rank, count


def _two_sided(rank: pd.DataFrame, count: pd.Series, k: pd.Series) -> pd.DataFrame:
    """Top k long, bottom k short, each side summing to 0.5 (nothing where k is 0)."""
    side = (0.5 / k.where(k > 0)).fillna(0.0)
    longs = rank.le(k, axis=0)
    shorts = rank.gt(count - k, axis=0)
    return longs.mul(side, axis=0) - shorts.mul(side, axis=0)


def _one_sided(rank: pd.DataFrame, k: pd.Series) -> pd.DataFrame:
    """Top k long in equal parts summing to 1 (nothing where k is 0)."""
    part = (1.0 / k.where(k > 0)).fillna(0.0)
    return rank.le(k, axis=0).mul(part, axis=0)


def _fraction_count(count: pd.Series, fraction: float) -> pd.Series:
    """floor(count * fraction), robust to float noise (0.29 * 100 is 28.999999999999996)."""
    return np.floor(count * fraction + 1e-9).astype(int)


def long_short(scores: pd.DataFrame, eligible: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Long the best `top_n`, short the worst `top_n`; sides +0.5 / -0.5."""
    rank, count = _ranks(scores, eligible)
    return _two_sided(rank, count, np.minimum(top_n, count // 2))


def long_only(scores: pd.DataFrame, eligible: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Long the best `top_n` equally, summing to 1."""
    rank, count = _ranks(scores, eligible)
    return _one_sided(rank, np.minimum(top_n, count))


def long_short_quantile(scores: pd.DataFrame, eligible: pd.DataFrame, *,
                        quantiles: int) -> pd.DataFrame:
    """Long the top group, short the bottom group, k = eligible // quantiles coins
    each; sides +0.5 / -0.5."""
    rank, count = _ranks(scores, eligible)
    return _two_sided(rank, count, count // quantiles)


def long_only_quantile(scores: pd.DataFrame, eligible: pd.DataFrame, *,
                       quantiles: int) -> pd.DataFrame:
    """Long the top group, k = eligible // quantiles coins, equally, summing to 1."""
    rank, count = _ranks(scores, eligible)
    return _one_sided(rank, count // quantiles)


def long_short_fraction(scores: pd.DataFrame, eligible: pd.DataFrame, *,
                        fraction: float) -> pd.DataFrame:
    """Long the top `fraction` of the eligible coins, short the bottom `fraction`,
    k = floor(eligible * fraction) each; sides +0.5 / -0.5."""
    rank, count = _ranks(scores, eligible)
    return _two_sided(rank, count, _fraction_count(count, fraction))


def long_only_fraction(scores: pd.DataFrame, eligible: pd.DataFrame, *,
                       fraction: float) -> pd.DataFrame:
    """Long the top `fraction` of the eligible coins, k = floor(eligible * fraction),
    equally, summing to 1."""
    rank, count = _ranks(scores, eligible)
    return _one_sided(rank, _fraction_count(count, fraction))


def spot_perp_carry(scores: pd.DataFrame, eligible: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Funding harvest, a coin against its perpetual: PERPETUAL-column weights of the
    selected coins; `Composed` adds the opposite weight in each coin's spot column.

    Selected are the `top_n` LOWEST scores (k = min(top_n, eligible) coins, none if no
    coin is eligible): the coins the signal would SELL on the perpetual. The perpetual
    leg gets -0.5 / k and the spot leg +0.5 / k, so the spot side sums to +0.5, the
    perpetual side to -0.5, gross 1 and net 0 (market-neutral per coin). The short
    perpetual receives the funding paid by longs when the rate is positive and pays it
    when negative: that is the carry.

    Why the lowest and not the highest: the `funding` signal is minus the trailing mean
    funding, so a coin with HIGH funding (most paid to the short) has a LOW score, which
    is exactly the coin to short the perpetual of. Taking the top scores here would
    select the coins whose short PAYS the most. With `momentum`/`reversal` the same rule
    means "the coins that signal would short"; the block makes no claim of funding then.
    Ties go to the coin listed first in the panel as in `_ranks`, so on a tie the LAST
    listed coins are the ones selected. Eligibility (spot tradeable and priced at the
    bar) is applied by `Composed` before this runs.
    """
    rank, count = _ranks(scores, eligible)
    k = np.minimum(top_n, count)
    part = (0.5 / k.where(k > 0)).fillna(0.0)
    return -rank.gt(count - k, axis=0).mul(part, axis=0)


def time_series_sign(scores: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Sign of each coin's own score, divided by the number of eligible coins."""
    masked = scores.where(eligible)
    count = masked.notna().sum(axis=1)
    part = (1.0 / count.where(count > 0)).fillna(0.0)
    return np.sign(masked).fillna(0.0).mul(part, axis=0)


PORTFOLIOS: dict[str, Block] = {
    "long_short": Block(long_short, {"top_n": positive_int}),
    "long_only": Block(long_only, {"top_n": positive_int}),
    "long_short_quantile": Block(long_short_quantile, {"quantiles": quantile_count(2)}),
    "long_only_quantile": Block(long_only_quantile, {"quantiles": quantile_count(1)}),
    "long_short_fraction": Block(long_short_fraction, {"fraction": fraction_up_to(0.5)}),
    "long_only_fraction": Block(long_only_fraction, {"fraction": fraction_up_to(1.0)}),
    "spot_perp_carry": Block(spot_perp_carry, {"top_n": positive_int}, needs_spot=True),
    "time_series_sign": Block(time_series_sign, {}),
}
