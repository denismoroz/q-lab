"""A direction forecast as a timing hint inside the live Bv2 book
(docs/research/forecast-in-book-protocol-2026-11-06.md).

Bv2 decides when to hedge; the hint only moves the moment: an entry the book wants waits until the
forecast sits low among its own past values (price more likely down) or a maximum wait passes; an
exit the book wants waits until the forecast sits high (price more likely up) or the wait passes.
`DelayGate` is that rule, one state per coin; `run_book_delay` runs frab's book with it (the
hedge-study driver with the gate swapped). The percentile reads only past forecasts (tested)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qlab import hedge_study as hs
from qlab.strategies.live import bv2 as _bv2


def past_percentile(f: np.ndarray, window: int = 2160, min_obs: int = 720) -> np.ndarray:
    """Per hour: the share of the previous `window` forecasts strictly below this one (NaN with
    fewer than `min_obs` past forecasts or no forecast)."""
    out = np.full(len(f), np.nan)
    s = pd.Series(f)
    # rank among the window including the current value, then remove the current one's own share
    n = s.rolling(window + 1, min_periods=min_obs + 1).count().to_numpy()
    r = s.rolling(window + 1, min_periods=min_obs + 1).rank(method="min").to_numpy()
    ok = ~np.isnan(r) & ~np.isnan(f) & (n > 1)
    out[ok] = (r[ok] - 1) / (n[ok] - 1)
    return out


@dataclass
class DelayGate:
    """The wish handed to the book under an entry and/or exit delay. `pct[i]`: the forecast's
    past percentile at bar i (NaN: no hint, the book's own wish passes)."""

    pct: np.ndarray
    entry: bool
    exit: bool
    q: float
    wait: int
    _on_wait: int = field(default=0, init=False)
    _off_wait: int = field(default=0, init=False)

    def step(self, own: bool, in_pos: bool, i: int) -> bool:
        p = self.pct[i]
        if own and not in_pos and self.entry:
            self._off_wait = 0
            go = np.isnan(p) or p <= self.q or self._on_wait >= self.wait
            self._on_wait = 0 if go else self._on_wait + 1
            return bool(go)
        if not own and in_pos and self.exit:
            self._on_wait = 0
            release = np.isnan(p) or p >= 1.0 - self.q or self._off_wait >= self.wait
            self._off_wait = 0 if release else self._off_wait + 1
            return not release
        self._on_wait = self._off_wait = 0
        return bool(own)


def delay_wish(own: np.ndarray, gate: DelayGate, start_in_pos: bool = False) -> np.ndarray:
    """The array version: the hedge is open after a bar exactly when the previous bar's handed
    wish was on (the book's rule without margin events)."""
    out = np.zeros(len(own), dtype=bool)
    in_pos = bool(start_in_pos)
    for i, w in enumerate(own):
        out[i] = gate.step(bool(w), in_pos, i)
        in_pos = out[i]
    return out


class _DelayDriver(hs._StudyDriver):
    def _gate(self, coin: str, own: bool, in_pos: bool, i: int) -> bool:
        return self._gates[coin].step(own, in_pos, i)


def run_book_delay(panel, params: Mapping[str, object],
                   gates: Mapping[str, DelayGate] | None = None) -> hs.BookRecord:
    """`hedge_study.run_book` with a `DelayGate` per coin (none: the production book)."""
    p = _bv2.B2Params.from_dict(dict(params))
    _bv2._require_frab_support(dict(params))
    driver = _DelayDriver(panel, p, None, gates=dict(gates or {}))
    run = driver.go()
    index, coins = panel.prices.index, list(p.coins)

    def frame(a):
        return pd.DataFrame(a, index=index, columns=coins)

    return hs.BookRecord(run=run, stepped=frame(driver.rec_stepped),
                         in_pos=frame(driver.rec_in_pos), own=frame(driver.rec_own),
                         hedge_acc=frame(driver.rec_acc), liquidated=frame(driver.rec_liq),
                         counts=dict(run.counts),
                         parts={k: frame(v) for k, v in driver.rec_parts.items()})


__all__ = ["DelayGate", "delay_wish", "past_percentile", "run_book_delay"]
