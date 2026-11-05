"""The hedge-trigger study reads the book's own hedged state, labels events only from bars after
them, and chooses thresholds only on the first 2/3
(docs/research/hedge-trigger-protocol-2026-11-04.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import hedge_study as hs
from qlab.strategies.live import _bv2_testkit as kit
from qlab.strategies.live.bv2 import LiveBv2

COINS = ("A", "B")


def _choppy(hours: int = 24 * 150, seed: int = 3) -> dict[str, np.ndarray]:
    """Price paths that turn often, so the book switches its hedge many times."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    paths = {}
    for k, c in enumerate(COINS):
        wave = 0.25 * np.sin(2 * np.pi * t / (24 * (20 + 9 * k)))
        paths[c] = 100.0 * np.exp(wave + np.cumsum(rng.normal(0, 0.004, hours)))
    return paths


def _market(hours: int = 24 * 150):
    return kit.market(coins=COINS, hours=hours, paths=_choppy(hours))


def test_the_recorded_hedge_is_the_book_s_own_and_the_run_is_unchanged() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    rec = hs.run_book(panel, params)
    live = LiveBv2().run(panel, params)
    pd.testing.assert_frame_equal(rec.run.weights, live.weights)
    pd.testing.assert_series_equal(rec.run.book_equity, live.book_equity)
    for c in COINS:
        stepped = rec.stepped[c]
        assert stepped.sum() > 0
        # the hedge is open exactly where the book holds a short of the perp
        assert (rec.in_pos[c][stepped] == (live.weights[c][stepped] < 0)).all()
        ev = hs.events(rec, c)
        assert (ev["kind"] == "on").sum() >= 3
        # events alternate on/off and sit on the bars where the state flips
        kinds = ev["kind"].tolist()
        assert all(a != b for a, b in zip(kinds, kinds[1:], strict=False))
        flips = rec.in_pos[c][stepped].astype(int).diff().fillna(0) != 0
        assert list(ev["fill"]) == list(flips[flips].index)
        assert (ev["decision"] < ev["fill"]).all()


def test_a_gate_that_vetoes_nothing_is_the_production_book() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    none = np.zeros(len(panel.prices.index), dtype=bool)
    rec = hs.run_book(panel, params, gates={c: (none, none) for c in COINS})
    pd.testing.assert_frame_equal(rec.run.weights, LiveBv2().run(panel, params).weights)


@pytest.mark.parametrize("which", ["on", "off"])
def test_the_vectorised_gate_equals_the_gate_on_the_live_book(which: str) -> None:
    panel, params = _market(), kit.params(coins=COINS)
    rng = np.random.default_rng(11)
    n = len(panel.prices.index)
    veto = {c: rng.random(n) < 0.5 for c in COINS}
    gates = {c: (veto[c], None) if which == "on" else (None, veto[c]) for c in COINS}
    rec = hs.run_book(panel, params, gates=gates)
    base = hs.run_book(panel, params)
    changed = False
    for c in COINS:
        st = rec.stepped[c].to_numpy()
        own = rec.own[c].to_numpy()[st]
        v = veto[c][st]
        first = bool(rec.in_pos[c].to_numpy()[st][0])
        final = hs.gate_wish(own, v if which == "on" else None, v if which == "off" else None,
                             start_in_pos=first)
        expected = np.r_[first, final[:-1]]
        assert (rec.in_pos[c].to_numpy()[st] == expected).all()
        changed |= bool((rec.in_pos[c] != base.in_pos[c]).any())
        # the book's own wish is not touched by the gate
        assert (rec.own[c] == base.own[c]).all()
    assert changed


def test_the_gate_skips_a_vetoed_on_and_holds_through_a_vetoed_off() -> None:
    own = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    veto = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert hs.gate_wish(own, veto, None).tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    hold = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert hs.gate_wish(own, None, hold).tolist() == [0, 1, 1, 1, 1, 1, 1, 0, 0]


def test_a_label_reads_only_bars_after_the_fill() -> None:
    index = pd.date_range("2026-01-01", periods=24 * 40, freq="1h", tz="UTC")
    spot = pd.Series(np.linspace(100, 140, len(index)), index=index)
    ev = pd.DataFrame({"fill": [index[100]], "decision": [index[99]], "kind": ["on"],
                       "liquidation": [False]})
    base = hs.label_events(ev, spot)
    before = spot.copy()
    before.iloc[:100] *= 3.0  # bars before the fill
    beyond = spot.copy()
    beyond.iloc[100 + 14 * 24 + 1:] *= 0.1  # bars after the longest label
    for other in (before, beyond):
        pd.testing.assert_frame_equal(hs.label_events(ev, other), base)
    seven = spot.copy()
    seven.iloc[100 + 7 * 24] = 1.0
    assert hs.label_events(ev, seven)["ret_7d"].iloc[0] < -0.9
    assert base["ret_7d"].iloc[0] == pytest.approx(spot.iloc[268] / spot.iloc[100] - 1)
    late = ev.assign(fill=[index[-10]])
    assert np.isnan(hs.label_events(late, spot)["ret_1d"].iloc[0])


def test_features_do_not_read_bars_after_the_one_they_are_written_at() -> None:
    panel = kit.market(coins=("A",), hours=24 * 130, paths=_choppy(24 * 130))
    perp, spot = panel.prices["A"], panel.prices["A-SPOT"] * 1.001
    funding = pd.Series(np.random.default_rng(1).normal(0, 1e-4, len(perp)), index=perp.index)
    volume = panel.volume["A"]
    full = hs.features(perp, spot, funding, volume, perp * 0.5)
    cut = 24 * 110 + 7
    head = hs.features(perp.iloc[:cut], spot.iloc[:cut], funding.iloc[:cut], volume.iloc[:cut],
                       (perp * 0.5).iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], head)
    assert full["vol_pct"].iloc[:24 * 90].isna().all()  # fewer than 90 past days: unknown
    assert full["vol_pct"].iloc[cut - 1] == full["vol_pct"].iloc[cut - 1]


def test_the_choice_reads_only_the_first_two_thirds() -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    cut = hs.split_date(index[0], index[-1])
    assert cut == pd.Timestamp("2020-07-18", tz="UTC")  # 299 days x 2/3 = 199.3
    a = pd.Series(np.where(index < cut, 1.0, -5.0), index=index)
    b = pd.Series(np.where(index < cut, 0.5, 9.0), index=index)
    assert hs.choose({"a": a, "b": b}, cut)[0] == "a"
    b_late = b.where(index < cut, 1e9)
    assert hs.choose({"a": a, "b": b_late}, cut)[0] == "a"
    assert hs.choose({"a": a, "b": b}, cut, eligible={"a": False, "b": True})[0] == "b"


def test_a_trigger_passes_the_book_s_sticky_exit() -> None:
    raw = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    priced = np.ones(len(raw), dtype=bool)
    wish = hs.trigger_wish(raw, priced, 3)
    assert wish.tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    assert hs.hedged_from_wish(wish).tolist() == [0, 0, 1, 1, 1, 0, 1, 1, 1, 0]


def test_hedge_money_is_the_short_s_return_funding_and_switch_cost() -> None:
    perp = np.array([100.0, 110.0, 99.0, 99.0])
    funding = np.array([0.0, 0.001, 0.0, 0.0])
    hedged = np.array([1, 1, 0, 0], dtype=bool)
    m = hs.hedge_money(perp, funding, hedged, 0.001)
    assert m[0] == pytest.approx(-0.10 + 0.001 - 0.001)
    assert m[1] == pytest.approx(0.10)
    assert m[2] == pytest.approx(-0.001)


def test_the_declared_grid_has_116_filters_and_53_triggers() -> None:
    variants = hs.declared_variants()
    assert sum(v.kind != "trigger" for v in variants) == 116
    assert sum(v.kind == "trigger" for v in variants) == 53
    assert len({v.name for v in variants}) == len(variants)


def test_the_false_share_counts_an_on_without_a_fall_as_false() -> None:
    ev = pd.DataFrame({"kind": ["on", "on", "on", "off", "off"],
                       "liquidation": [False, False, False, False, True],
                       "ret_7d": [-0.1, 0.0, 0.2, 0.3, -0.5]})
    assert hs.false_share(ev, "on") == (pytest.approx(2 / 3), 3)
    assert hs.false_share(ev, "off") == (0.0, 1)


def test_a_share_handed_to_the_book_sizes_its_hedge() -> None:
    panel = _market()
    params = kit.params(coins=COINS, partial_hedge_share=0.5)
    n = len(panel.prices.index)
    half = hs.run_book(panel, params, shares={c: np.full(n, 0.5) for c in COINS})
    full = hs.run_book(panel, params, shares={c: np.full(n, 1.0) for c in COINS})
    for c in COINS:
        for rec, share in ((half, 0.5), (full, 1.0)):
            w = rec.run.weights
            on = rec.in_pos[c] & rec.stepped[c]
            covered = (-w[c][on] / w[f"{c}-SPOT"][on]).dropna()  # short / spot, same price
            assert len(covered) > 100
            assert covered.median() == pytest.approx(share, abs=0.05)
    with pytest.raises(ValueError, match="partial_hedge_share"):
        hs.run_book(panel, kit.params(coins=COINS), shares={c: np.full(n, 0.5) for c in COINS})


def test_episode_parts_add_up_to_the_episode_money() -> None:
    panel, params = _market(), kit.params(coins=COINS)
    rec = hs.run_book(panel, params)
    for c in COINS:
        ep = hs.episodes(rec, c)
        ev = hs.events(rec, c)
        on = ev[(ev["kind"] == "on") & ev["money"].notna()]
        assert len(ep) == len(on) > 0
        assert np.allclose(ep["money"].to_numpy(), on["money"].to_numpy())
        assert np.allclose(ep["money"], ep["price"] + ep["funding"] - ep["fees"])
        assert (ep["fees"] > 0).all() and (ep["notional"] > 0).all() and (ep["hours"] > 0).all()
