"""Follow-up of the hedge-trigger study: the band wish at zero is the live book's own wish, the
momentum measures read the right bars, the bootstrap keeps whole days together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import hedge_followup as hf
from qlab import hedge_study as hs
from qlab.strategies.live import _bv2_testkit as kit
from qlab.test_hedge_study import COINS, _market


def test_the_band_wish_with_every_knob_at_zero_is_the_book_s_own_wish() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    rec = hs.run_book(panel, params)
    for c in COINS:
        st = rec.stepped[c].to_numpy()
        wish = hf.band_wish(panel.prices[c].to_numpy(dtype=float))
        assert ((wish[st] > 0.5) == rec.own[c].to_numpy()[st]).all()


def test_the_band_wish_fed_to_the_book_is_the_production_book() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    wishes = {c: hf.band_wish(panel.prices[c].to_numpy(dtype=float)) for c in COINS}
    rec = hs.run_book(panel, params, wishes=wishes)
    pd.testing.assert_frame_equal(rec.run.weights, hs.run_book(panel, params).run.weights)


def test_a_wider_band_and_a_pause_switch_less() -> None:
    rng = np.random.default_rng(4)
    perp = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, 24 * 300)))

    def switches(w):
        w = np.nan_to_num(w)
        return int(np.abs(np.diff(w)).sum())

    base = switches(hf.band_wish(perp))
    assert switches(hf.band_wish(perp, 0.05, 0.05)) < base
    assert switches(hf.band_wish(perp, min_hold=168, cooldown=168)) < base


def test_min_hold_keeps_the_hedge_and_cooldown_blocks_a_new_one() -> None:
    hours = 24 * 40
    up = 100 * np.exp(np.r_[np.linspace(0, 0.3, 24 * 31), np.linspace(0.3, 0.0, hours - 24 * 31)])
    perp = np.r_[up]
    w0 = np.nan_to_num(hf.band_wish(perp))
    w1 = np.nan_to_num(hf.band_wish(perp, min_hold=10**6))
    assert w1.sum() >= w0.sum()
    assert (w1[np.argmax(w1 > 0):] == 1).all()  # once on, never off


def test_momentum_pairs_read_the_past_before_and_the_next_after_the_day() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series(np.exp(np.arange(10.0)), index=idx)
    p = hf.momentum_pairs(close, 3, 2)
    assert p["past"].iloc[5] == pytest.approx(3.0)
    assert p["next"].iloc[5] == pytest.approx(2.0)
    assert np.isnan(p["next"].iloc[-1]) and np.isnan(p["past"].iloc[0])


def test_a_random_walk_has_no_momentum() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=3000, freq="D")
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.03, 3000))), index=idx)
    assert hf.variance_ratio(close, 14) == pytest.approx(1.0, abs=0.25)
    p = hf.momentum_pairs(close, 14, 14).dropna()
    x, y = p["past"].to_numpy(), p["next"].to_numpy()
    point, lo, hi = hf.block_bootstrap(p.index.to_numpy(), lambda r: hf.corr(x[r], y[r]),
                                       draws=200)
    assert lo < 0 < hi
    assert abs(hf.sign_excess(x, y)) < 0.06


def test_trending_prices_show_momentum() -> None:
    idx = pd.date_range("2018-01-01", periods=2000, freq="D")
    rng = np.random.default_rng(1)
    drift = np.repeat(rng.choice([-0.02, 0.02], size=40), 50)
    close = pd.Series(np.exp(np.cumsum(drift + rng.normal(0, 0.01, 2000))), index=idx)
    assert hf.variance_ratio(close, 14) > 1.5
    p = hf.momentum_pairs(close, 14, 7).dropna()
    assert hf.corr(p["past"].to_numpy(), p["next"].to_numpy()) > 0.3


def test_max_drawdown_counts_the_fall_from_the_start() -> None:
    assert hf.max_drawdown(np.array([-0.1, 0.05, 0.2, -0.3, 0.1])) == pytest.approx(-0.3)
    assert hf.max_drawdown(np.array([-0.1, -0.1])) == pytest.approx(-0.2)
