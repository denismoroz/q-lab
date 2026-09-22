"""TSMOM ensemble on crypto perps -- driven by CALLING the live engine's own
code (`frab.strategy.trend.signals`), not by re-deriving it.

Contrast with `qlab.strategies.trend.TrendTSMOMEnsemble`: that module is a
transcription, and its own docstring names exactly what it leaves out --
`book_vol_scale`, "the only knob of seven that helped on BOTH the fitted and
the never-seen window", because a memoryless `target_weights(panel, params)
-> weights` function called once has no realised equity curve to scale by.
`docs/TASKS.md` T29's 2026-09-22 addendum answers that: the memorylessness
was q-lab's OWN interface decision (T24), not a fact about the strategy, and
the live functions are pure enough to drive bar by bar, feeding each one
back the equity it would actually have realised so far. This module does
exactly that -- it does not carry its own copy of the signal, sizing, or
book-volatility-targeting logic; it calls
`frab.strategy.trend.signals.target_weights` and `.book_vol_scale` once per
bar through `qlab.strategies.live._loader.import_frab`.

Params come from `frab.strategy.trend.params.TrendParams.from_dict` -- not
hand-mapped -- so a spec's `params:` block is validated by frab's own rules
(positive lookbacks, `mode == "paper"`, etc), not by a second, possibly
drifted copy of them here.

## The causal book-equity feed -- the highest-risk part of this module

`book_vol_scale(daily_equity, params)` scales the whole book by a factor
computed from the BOOK'S OWN trailing realised equity curve (frab's
docstring: "`daily_equity` ... ending at the last closed day (causal)").
This driver builds that curve itself, bar by bar, from the weights it has
ALREADY decided and the price/funding history already available -- and the
one rule that must never be broken is:

    when deciding the weight for bar `t`, the equity series handed to
    `book_vol_scale` must end at bar `t-1`'s REALISED return AT THE
    LATEST -- never at `t`'s.

Bar `t`'s own realised return (the interval `(t-1, t]`, earned by the
weight decided at `t-1`) is fully computable the moment bar `t`'s price is
known -- which is also the moment this driver is asked to decide bar `t`'s
OWN weight, from closes "up to and including `t`" (the same causal boundary
`qlab.strategies.trend`'s transcription already uses). It would therefore
be easy, and WRONG, to fold that return into the equity curve before
calling `book_vol_scale` for bar `t` -- using it to size the very bar whose
price move produced it. That is look-ahead with a book-volatility mechanism
in the loop, which is exactly the failure mode this project has hit four
times (`docs/TASKS.md` T29): a huge move at `t` would immediately shrink or
grow bar `t`'s own weight through the scale factor, producing a beautiful,
wrong result. See `qlab.strategies.live.test_trend` for the adversarial
test this rule earns, and its docstring for the deliberate re-break that
was performed to confirm the test actually catches the bug.

Concretely, per bar `i` (0-indexed row of `panel.prices`):

  1. `book_vol_scale` is called with the equity curve built from bars
     `0 .. i-1` only (bar `i`'s own realised return has not been appended
     yet).
  2. `target_weights` is called with `closes_by_coin` built from prices
     `0 .. i` (this IS causally fine -- signal and per-asset vol sizing use
     "up to and including `t`", same as the transcription).
  3. ONLY AFTER both calls return, this bar's realised return (weight
     decided at `i-1`, applied to the `(i-1, i]` price move and funding) is
     computed and appended to the equity curve, for bar `i+1` to see.

## What this does NOT reproduce, and why

The realised-equity proxy fed to `book_vol_scale` is gross price return
plus funding accrual (via `qlab.harness.accrual.compute_accrual`, so it
inherits that function's sign convention and its refusal to treat unknown
funding as zero) -- it does NOT include trading costs (fees/slippage).
`qlab.harness.strategy.Strategy.target_weights(panel, params)` is not
handed a `CostModel`; inventing a fee number here to fold into the equity
curve would be exactly the kind of fabricated parameter `CLAUDE.md` forbids
("Никогда не придумывай число"). This is a documented fidelity gap, not a
look-ahead risk: omitting a real, small, deterministic drag cannot smuggle
future information into a bar's decision, it can only make the book read
very slightly less volatile than the live one (fees are a few basis points
per unit of turnover, dwarfed by daily price and funding moves for a
directional trend book).

Coins named in `params.coins` that are absent from `panel.prices` simply
never trade (their weight is 0 for lack of a column to hold it) -- this
mirrors the live engine's own "a coin without enough history gets weight 0"
rule for insufficient history, extended to "no data at all".
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.data.snapshot import _top_k_by_volume_mask
from qlab.harness.accrual import compute_accrual
from qlab.harness.capacity import _bar_interval
from qlab.harness.panel import MarketPanel
from qlab.strategies.live._loader import import_frab

_signals = import_frab("frab.strategy.trend.signals")
_params_module = import_frab("frab.strategy.trend.params")
TrendParams = _params_module.TrendParams


class LiveTrendTSMOMEnsemble:
    """Drives `frab.strategy.trend.signals.target_weights` bar by bar, with
    a causal, self-computed book-volatility-targeting feed. See module
    docstring for the full contract and the look-ahead guard it relies on.

    `params` is the spec's `params:` mapping, handed unmodified to
    `TrendParams.from_dict` -- every key `TrendParams` knows about
    (`coins`, `lookbacks`, `vol_target_daily`, `book_vol_target_ann`, ...)
    is accepted; unknown keys are silently ignored by `from_dict` itself
    (not by this class), matching frab's own contract for its config
    loader.
    """

    name = "trend-live"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        raw_params = dict(params)
        # `coins: null` means "every instrument this panel has", the same
        # convention `specs/trend.yaml` and `specs/xsmom.yaml` already use
        # for an unrestricted universe. The live `TrendParams` cannot hold
        # it -- its own validator rejects an empty coin list, because in
        # production the list is always a deliberate choice -- so the panel's
        # columns are substituted here, before the live dataclass is built.
        #
        # This is what makes the point-in-time liquidity experiment possible:
        # with `coins: null` the candidate set is the whole discovered
        # universe and `data.min_daily_volume_usd` decides, per bar, which of
        # them were liquid enough BEFORE that bar -- as opposed to a fixed
        # list written on one date and applied backwards (docs/T29_LIVE_TREND.md).
        if raw_params.get("coins") is None:
            raw_params["coins"] = list(panel.prices.columns)
        trend_params = TrendParams.from_dict(raw_params)

        prices = panel.prices
        index = prices.index
        columns = prices.columns
        no_funding_instruments = panel.meta.get("no_funding_instruments", ())

        # Only coins the live params actually name AND that this panel has
        # a column for can ever get a non-zero weight -- see module
        # docstring's "what this does not reproduce" for coins named but
        # absent from the panel.
        traded_coins = [c for c in trend_params.coins if c in columns]

        # `top_k_by_volume` — the point-in-time counterpart of a hand-written
        # coin list. The live engine's `coins` IS a strategy parameter, so its
        # honest analogue belongs here rather than in the snapshot: at each
        # bar the candidate set becomes the k most traded instruments by
        # trailing 24h USD volume observed strictly BEFORE that bar.
        #
        # k is held FIXED on purpose. A plain volume threshold lets the
        # surviving count drift with the market's overall liquidity, and book
        # breadth is not a neutral detail — the same idea on 20 legs and on
        # 148 behaved as two different strategies (docs/XSMOM_T21.md), and the
        # $1M threshold run held a median of 53 legs against the live list's
        # 25. Comparing a 25-leg book with a 53-leg one measures breadth and
        # selection at once and therefore measures neither. With k fixed at
        # the live list's own length, exactly one thing differs between the
        # two runs: WHEN the coins were chosen.
        top_k = raw_params.get("top_k_by_volume")
        rank_mask: pd.DataFrame | None = None
        if top_k is not None:
            if panel.volume is None or bool(panel.volume.isna().all().all()):
                raise ValueError(
                    "top_k_by_volume needs a panel carrying volume; this snapshot has none "
                    "(unknown volume is not zero volume — rebuild the snapshot)"
                )
            rank_mask = _top_k_by_volume_mask(
                prices, panel.volume, _bar_interval(index), int(top_k)
            )

        running_closes: dict[str, list[float]] = {c: [] for c in traded_coins}
        weight_rows: list[dict[str, float]] = []

        # `equity_hist[k]` is the book's cumulative realised equity AFTER
        # the return that completed at row `k` (k >= 1); `equity_hist[0]`
        # is the baseline before any return exists. Ratios are all that
        # matter to `book_vol_scale`, so the baseline value is arbitrary;
        # 1.0 is chosen for readability only.
        equity_hist: list[float] = [1.0]
        prev_weights: dict[str, float] | None = None
        prev_row_prices: pd.Series | None = None

        for i, ts in enumerate(index):
            row_prices = prices.iloc[i]

            for coin in traded_coins:
                value = row_prices.get(coin)
                if pd.notna(value):
                    running_closes[coin].append(float(value))

            # --- Step 1: book_vol_scale on bars [0, i-1] only. Do NOT touch
            # equity_hist between here and step 2 -- see module docstring.
            scale = _signals.book_vol_scale(equity_hist, trend_params)

            # --- Step 2: target_weights on closes through bar i (causally
            # fine -- same boundary the transcription uses).
            closes_by_coin = {coin: running_closes[coin] for coin in traded_coins}
            raw_weights = _signals.target_weights(closes_by_coin, trend_params, size_scale=scale)

            tradeable_row = panel.tradeable.iloc[i]
            rank_row = None if rank_mask is None else rank_mask.iloc[i]
            weights_i = {
                coin: (
                    raw_weights.get(coin, 0.0)
                    if bool(tradeable_row.get(coin, False))
                    and (rank_row is None or bool(rank_row.get(coin, False)))
                    else 0.0
                )
                for coin in traded_coins
            }
            weight_rows.append(weights_i)

            # --- Step 3: NOW extend the equity curve with the return that
            # completes AT this bar, using the PREVIOUS bar's weight -- this
            # becomes visible starting at bar i+1's decision, never at this
            # bar's own.
            if prev_weights is not None and prev_row_prices is not None:
                gross_ret = 0.0
                for coin, w in prev_weights.items():
                    if w == 0.0:
                        continue
                    p0 = prev_row_prices.get(coin)
                    p1 = row_prices.get(coin)
                    if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                        gross_ret += w * (p1 / p0 - 1.0)

                prev_weights_frame = pd.DataFrame([prev_weights], index=[ts]).reindex(
                    columns=columns, fill_value=0.0
                )
                funding_row = panel.funding.loc[[ts]]
                accrual_ret = float(
                    compute_accrual(
                        prev_weights_frame,
                        funding_row,
                        no_funding_instruments=no_funding_instruments,
                    ).iloc[0]
                )

                equity_hist.append(equity_hist[-1] * (1.0 + gross_ret + accrual_ret))

            prev_weights = weights_i
            prev_row_prices = row_prices

        weights = pd.DataFrame(weight_rows, index=index)
        weights = weights.reindex(columns=columns, fill_value=0.0)
        # Defensive re-assertion of the per-bar tradeable mask already
        # applied above -- cheap, and makes the invariant checkable in one
        # place even if a future edit changes the loop.
        weights = weights.where(panel.tradeable, 0.0)
        return weights
