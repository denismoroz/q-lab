"""Tests for the overlay mechanism and the shipped overlays (docs/OVERLAYS.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.lookahead import lookahead_violation
from qlab.harness.panel import MarketPanel
from qlab.strategies.overlays import Overlaid, core, get, names, parse_chain
from qlab.strategies.overlays.book import MARKET_GATE, OWN_RESULTS, VOL_TARGET
from qlab.strategies.overlays.core import BookOverlay, Context, register
from qlab.strategies.overlays.position import GRADUAL_ENTRY, GRADUAL_EXIT, ladder
from qlab.strategies.regime_gate import RegimeGate

INNER = "qlab.strategies.overlays.test_overlays:Momentum"
COSTS = {"taker_fee_bps": 3.5, "slippage_bps": 0.9}


class Momentum:
    """Test strategy: long the coins that rose over `lookback` bars, short the ones that fell,
    equal sized, decided every bar from the past only."""

    name = "momentum"

    def target_weights(self, panel, params):
        lookback = int(params.get("lookback", 10))
        sign = np.sign(panel.prices / panel.prices.shift(lookback) - 1.0).fillna(0.0)
        weights = sign / panel.prices.shape[1]
        return weights.where(panel.tradeable, 0.0)


class Constant:
    name = "constant"

    def target_weights(self, panel, params):
        return pd.DataFrame(float(params.get("w", 0.5)), index=panel.prices.index,
                            columns=panel.prices.columns)


def make_panel(rows: int = 500, cols: int = 6, *, seed: int = 0, freq: str = "1D",
               snapshot: str = "syn") -> MarketPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=rows, freq=freq, tz="UTC")
    names_ = [f"C{i}" for i in range(cols)]
    steps = rng.normal(0.0004, 0.025, (rows, cols))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=names_)
    return MarketPanel(snapshot_id=snapshot, prices=prices,
                       funding=pd.DataFrame(0.0, index=index, columns=names_),
                       tradeable=pd.DataFrame(True, index=index, columns=names_), meta={})


def flat_panel(prices: list[float], *, cols: int = 1) -> MarketPanel:
    index = pd.date_range("2023-01-01", periods=len(prices), freq="1D", tz="UTC")
    frame = pd.DataFrame({f"C{i}": prices for i in range(cols)}, index=index)
    return MarketPanel(snapshot_id=f"flat-{prices[:3]}", prices=frame,
                       funding=pd.DataFrame(0.0, index=index, columns=frame.columns),
                       tradeable=pd.DataFrame(True, index=index, columns=frame.columns), meta={})


def overlaid(chain: list[dict], inner: str = INNER, inner_params: dict | None = None) -> dict:
    return {"inner_code_ref": inner, "inner_params": inner_params or {}, "overlays": chain,
            "costs": COSTS}


@pytest.fixture(autouse=True)
def _btc_labels(monkeypatch):
    """The market gate reads BTC closes from outside the panel: a fixed synthetic series."""
    index = pd.date_range("2021-06-01", periods=1500, freq="1D", tz="UTC")
    rng = np.random.default_rng(7)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(index)))), index=index)
    monkeypatch.setattr("qlab.strategies.detectors.market_closes", lambda: closes)


# --------------------------------------------------------------------------
# The mechanism


def test_the_shipped_overlays_are_registered_at_both_levels() -> None:
    assert {"market_gate", "meta_gate", "vol_target", "own_results", "gradual_entry",
            "gradual_exit"} <= set(names())
    assert {get(n).level for n in names()} == {"book", "position"}


def test_a_name_is_taken_once_and_an_unknown_one_is_named() -> None:
    with pytest.raises(ValueError, match="registered already"):
        register(VOL_TARGET)
    with pytest.raises(ValueError, match="unknown overlay 'nope'"):
        parse_chain([{"name": "nope"}])


def test_a_typo_in_a_parameter_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_chain([{"name": "vol_target", "params": {"lookback_days": 30, "max_scale": 1.0,
                                                       "lookbak": 3}}])
    with pytest.raises(ValueError, match="missing"):
        parse_chain([{"name": "vol_target", "params": {"lookback_days": 30}}])


def test_an_empty_chain_is_the_inner_strategy_and_every_param_is_required() -> None:
    panel = make_panel()
    inner = Momentum().target_weights(panel, {})
    out = Overlaid().target_weights(panel, overlaid([]))
    pd.testing.assert_frame_equal(out, inner)
    with pytest.raises(ValueError, match="needs params"):
        Overlaid().target_weights(panel, {"inner_code_ref": INNER, "inner_params": {}})


def test_overlays_that_read_results_need_the_costs() -> None:
    panel = make_panel()
    chain = [{"name": "own_results", "params": OWN_RESULTS.grid()["w30-f0-step-up1"]}]
    params = overlaid(chain)
    del params["costs"]
    with pytest.raises(ValueError, match="costs"):
        Overlaid().target_weights(panel, params)


class _Plus(BookOverlay):
    name = "toy_plus"
    short = "plus"
    params = ("x",)

    def scale(self, ctx, params):
        return pd.Series(1.0, index=ctx.index)

    def apply(self, weights, ctx, params):
        return weights + float(params["x"])

    def grid(self):
        return {}


class _Times(BookOverlay):
    name = "toy_times"
    short = "times"
    params = ("x",)

    def scale(self, ctx, params):
        return pd.Series(float(params["x"]), index=ctx.index)

    def grid(self):
        return {}


@pytest.fixture
def toys(monkeypatch):
    monkeypatch.setitem(core._REGISTRY, "toy_plus", _Plus())
    monkeypatch.setitem(core._REGISTRY, "toy_times", _Times())


def test_overlays_apply_in_the_declared_order(toys) -> None:
    panel = make_panel(rows=20)
    plus_then_times = overlaid([{"name": "toy_plus", "params": {"x": 1.0}},
                                {"name": "toy_times", "params": {"x": 2.0}}],
                               inner=f"{__name__}:Constant", inner_params={"w": 0.5})
    times_then_plus = overlaid(list(reversed(plus_then_times["overlays"])),
                               inner=f"{__name__}:Constant", inner_params={"w": 0.5})
    first = Overlaid().target_weights(panel, plus_then_times)
    second = Overlaid().target_weights(panel, times_then_plus)
    assert first.iloc[0, 0] == pytest.approx((0.5 + 1.0) * 2.0)
    assert second.iloc[0, 0] == pytest.approx(0.5 * 2.0 + 1.0)


def test_nothing_is_held_where_nothing_can_be_traded(toys) -> None:
    panel = make_panel(rows=20, cols=2)
    tradeable = panel.tradeable.copy()
    tradeable.iloc[5:, 1] = False
    panel = MarketPanel(snapshot_id="t", prices=panel.prices, funding=panel.funding,
                        tradeable=tradeable, meta={})
    out = Overlaid().target_weights(panel, overlaid(
        [{"name": "toy_plus", "params": {"x": 1.0}}], inner=f"{__name__}:Constant"))
    assert (out.iloc[5:, 1] == 0.0).all() and (out.iloc[5:, 0] != 0.0).all()


def test_the_inner_strategy_is_run_once_for_many_overlay_settings(monkeypatch) -> None:
    panel = make_panel(snapshot="cache-test")
    calls = []
    real = Momentum.target_weights

    def counting(self, p, params):
        calls.append(1)
        return real(self, p, params)

    monkeypatch.setattr(Momentum, "target_weights", counting)
    for point in list(OWN_RESULTS.grid().values())[:5]:
        Overlaid().target_weights(panel, overlaid([{"name": "own_results", "params": point}]))
    assert len(calls) == 1


def test_diagnostics_report_the_share_of_exposure_left() -> None:
    panel = make_panel()
    strategy = Overlaid()
    strategy.target_weights(panel, overlaid([{"name": "own_results", "params": {
        "window_days": 14, "floor": 0.0, "mode": "step", "up_days": 1}}]))
    assert 0.0 < strategy.diagnostics()["overlay_gross_ratio"] < 1.0


# --------------------------------------------------------------------------
# Look-ahead: every overlay, every point of its grid, through the pipeline's own guard


def _all_points():
    for name in names():
        overlay = get(name)
        for label, point in overlay.grid().items():
            yield pytest.param(name, point, id=f"{name}:{label}")


@pytest.mark.parametrize("name,point", list(_all_points()))
def test_no_overlay_reads_the_future_at_any_grid_point(name, point) -> None:
    panel = make_panel(rows=260, cols=4, snapshot=f"la-{name}")
    params = overlaid([{"name": name, "params": point}])
    strategy = Overlaid()
    weights = strategy.target_weights(panel, params)
    assert lookahead_violation(Overlaid(), panel, params, weights) is None


def test_the_look_ahead_guard_catches_an_overlay_that_reads_the_future(monkeypatch) -> None:
    """The test above has teeth: an overlay that uses the return of the bar it decides for
    (not shifted by one) is caught."""

    class Leaky(BookOverlay):
        name = "toy_leaky"
        short = "leak"
        params = ()
        needs_returns = True

        def scale(self, ctx, params):
            return (ctx.own_returns.reindex(ctx.index).fillna(0.0) > 0).astype(float)

        def grid(self):
            return {}

    monkeypatch.setitem(core._REGISTRY, "toy_leaky", Leaky())
    panel = make_panel(rows=260, cols=4, snapshot="leaky")
    params = overlaid([{"name": "toy_leaky", "params": {}}])
    weights = Overlaid().target_weights(panel, params)
    assert lookahead_violation(Overlaid(), panel, params, weights) is not None


# --------------------------------------------------------------------------
# Book level


def test_the_market_gate_equals_the_regime_gate_wrapper() -> None:
    panel = make_panel(rows=400, cols=4, snapshot="gate-eq")
    point = MARKET_GATE.grid()["skip-bull"]
    new = Overlaid().target_weights(panel, overlaid([{"name": "market_gate", "params": point}]))
    old = RegimeGate().target_weights(panel, {
        "inner_code_ref": INNER, "inner_params": {}, "trade_in": point["trade_in"]})
    pd.testing.assert_frame_equal(new, old)
    assert (new.abs().sum(axis=1) == 0).any()  # it did switch the book off somewhere


def _ctx(returns: list[float], panel: MarketPanel | None = None) -> Context:
    panel = panel or flat_panel([100.0] * (len(returns) + 1))
    series = pd.Series(returns, index=panel.prices.index[:len(returns)])
    return Context(panel=panel, own_returns=series)


def test_own_results_step_scales_down_after_losses_and_back_up_after_gains() -> None:
    returns = [0.01] * 10 + [-0.02] * 10 + [0.03] * 10
    ctx = _ctx(returns)
    point = {"window_days": 5, "floor": 0.25, "mode": "step", "up_days": 1}
    scale = OWN_RESULTS.scale(ctx, point)
    assert scale.iloc[8] == 1.0                    # gains in the window
    assert scale.iloc[17] == 0.25                  # five losing days known
    assert scale.iloc[26] == 1.0                   # five winning days known again
    assert scale.iloc[:5].eq(1.0).all()            # before the window is full: left alone


def test_own_results_smooth_is_proportional_and_clipped() -> None:
    rng = np.random.default_rng(3)
    returns = list(rng.normal(0.0, 0.01, 60)) + [0.01] * 40
    scale = OWN_RESULTS.scale(_ctx(returns), {
        "window_days": 30, "floor": 0.25, "mode": "smooth", "up_days": 1})
    assert scale.between(0.25, 1.0).all()
    assert scale.iloc[-1] == 1.0                   # a clear run of gains: full size
    assert 0.25 < scale.iloc[40:60].mean() < 1.0   # noise around zero: in between


def test_own_results_comes_back_up_gradually_when_asked() -> None:
    returns = [-0.02] * 10 + [0.05] * 30
    fast = OWN_RESULTS.scale(_ctx(returns), {
        "window_days": 5, "floor": 0.0, "mode": "step", "up_days": 1})
    slow = OWN_RESULTS.scale(_ctx(returns), {
        "window_days": 5, "floor": 0.0, "mode": "step", "up_days": 10})
    assert fast.iloc[-1] == 1.0 and slow.iloc[-1] == 1.0
    assert fast.iloc[9] == 0.0 and slow.iloc[9] == 0.0     # both went down at once
    first_fast = fast.iloc[10:][fast.iloc[10:] > 0.5].index[0]
    first_slow = slow.iloc[10:][slow.iloc[10:] > 0.5].index[0]
    assert first_slow > first_fast
    assert slow.diff().dropna().max() <= 0.1 + 1e-9        # at most a tenth of the range a bar


def test_own_results_reads_only_what_was_realised_before_the_bar() -> None:
    """A big loss at row t changes the scale from row t + 1, not at t."""
    returns = [0.01] * 20 + [-0.5] + [0.01] * 5
    scale = OWN_RESULTS.scale(_ctx(returns), {
        "window_days": 3, "floor": 0.0, "mode": "step", "up_days": 1})
    assert scale.iloc[20] == 1.0 and scale.iloc[21] == 0.0


def test_vol_target_shrinks_the_book_in_a_turbulent_stretch_and_never_exceeds_max_scale() -> None:
    rng = np.random.default_rng(5)
    calm = rng.normal(0.0, 0.005, 150)
    wild = rng.normal(0.0, 0.05, 30)
    ctx = _ctx(list(np.r_[calm, wild, calm[:60]]))
    scale = VOL_TARGET.scale(ctx, {"lookback_days": 10, "max_scale": 1.0})
    assert scale.iloc[170:180].max() < 0.5          # during the storm
    assert scale.max() <= 1.0
    bigger = VOL_TARGET.scale(ctx, {"lookback_days": 10, "max_scale": 2.0})
    assert bigger.max() > 1.0 and bigger.max() <= 2.0
    assert scale.iloc[:10].eq(1.0).all()            # no history yet: left alone


def test_vol_target_leaves_a_flat_book_alone() -> None:
    scale = VOL_TARGET.scale(_ctx([0.0] * 80), {"lookback_days": 10, "max_scale": 2.0})
    assert scale.eq(1.0).all()


def test_a_scale_that_is_not_a_number_is_refused() -> None:
    class Broken(BookOverlay):
        name = "toy_broken"
        short = "broken"
        params = ()

        def scale(self, ctx, params):
            return pd.Series(-1.0, index=ctx.index)

        def grid(self):
            return {}

    panel = make_panel(rows=10)
    with pytest.raises(ValueError, match="scale"):
        Broken().apply(Constant().target_weights(panel, {}), Context(panel, None), {})


# --------------------------------------------------------------------------
# Position level


def _walk(target: list[float], prices: list[float] | None = None, **kw) -> list[float]:
    prices = prices or [100.0] * len(target)
    panel = flat_panel(prices)
    targets = pd.DataFrame({"C0": target}, index=panel.prices.index)
    args = dict(tranches_in=1, gap_in=1, only_if_profit=False, tranches_out=1, gap_out=1)
    args.update(kw)
    return ladder(targets, panel.prices, panel.tradeable, **args)["C0"].round(6).tolist()


def test_a_new_position_is_taken_in_equal_parts() -> None:
    held = _walk([0, 0, 1, 1, 1, 1, 1, 1, 1], tranches_in=3, gap_in=2)
    assert held == pytest.approx([0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1], abs=1e-6)


def test_a_position_is_left_in_parts_and_a_closed_book_ends_at_zero() -> None:
    held = _walk([1, 1, 0, 0, 0, 0, 0, 0], tranches_out=3, gap_out=2)
    assert held == pytest.approx([1, 1, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0, 0], abs=1e-6)


def test_the_parts_follow_a_target_that_moves_and_the_last_one_reaches_it() -> None:
    held = _walk([0, 1, 1, 2, 2, 2, 2], tranches_in=2, gap_in=2)
    # first part of 1 -> 0.5; next part is due two bars later and takes half the gap to the
    # target AS IT IS THEN (2): 0.5 + 1.5 = 2.0.
    assert held[1] == pytest.approx(0.5) and held[3] == pytest.approx(2.0)


def test_adding_only_while_in_profit() -> None:
    target = [0, 1, 1, 1, 1, 1, 1, 1]
    falling = [100, 100, 99, 98, 97, 96, 95, 94]
    rising = [100, 100, 101, 102, 103, 104, 105, 106]
    stuck = _walk(target, falling, tranches_in=3, gap_in=2, only_if_profit=True)
    assert stuck[-1] == pytest.approx(1 / 3)           # never in profit: never added to
    grown = _walk(target, rising, tranches_in=3, gap_in=2, only_if_profit=True)
    assert grown[-1] == pytest.approx(1.0)
    plain = _walk(target, falling, tranches_in=3, gap_in=2, only_if_profit=False)
    assert plain[-1] == pytest.approx(1.0)             # without the test it adds anyway


def test_a_short_position_is_in_profit_when_the_price_falls() -> None:
    held = _walk([0, -1, -1, -1, -1, -1], [100, 100, 99, 98, 97, 96],
                 tranches_in=3, gap_in=2, only_if_profit=True)
    assert held[-1] == pytest.approx(-1.0)


def test_a_position_that_changes_side_is_closed_at_once_and_entered_gradually() -> None:
    held = _walk([1, 1, -1, -1, -1, -1], tranches_in=2, gap_in=2, tranches_out=2, gap_out=2)
    assert held[0] == pytest.approx(0.5)               # entering the long
    assert held[2] == pytest.approx(-0.5)              # long closed, short started
    assert held[4] == pytest.approx(-1.0)


def test_nothing_is_held_where_nothing_can_be_traded_even_while_leaving() -> None:
    panel = flat_panel([100.0] * 6)
    tradeable = panel.tradeable.copy()
    tradeable.iloc[3:, 0] = False
    targets = pd.DataFrame({"C0": [1, 1, 1, 0, 0, 0]}, index=panel.prices.index, dtype=float)
    held = ladder(targets, panel.prices, tradeable, tranches_in=1, gap_in=1, only_if_profit=False,
                  tranches_out=3, gap_out=2)["C0"]
    assert held.iloc[:3].tolist() == [1, 1, 1] and held.iloc[3:].eq(0.0).all()


def test_a_book_of_many_coins_moves_each_coin_on_its_own_clock() -> None:
    panel = flat_panel([100.0] * 8, cols=2)
    targets = pd.DataFrame({"C0": [1, 1, 1, 1, 1, 1, 1, 1], "C1": [0, 0, 1, 1, 1, 1, 1, 1]},
                           index=panel.prices.index, dtype=float)
    held = ladder(targets, panel.prices, panel.tradeable, tranches_in=2, gap_in=2,
                  only_if_profit=False, tranches_out=1, gap_out=1)
    assert held["C0"].tolist() == pytest.approx([0.5, 0.5, 1, 1, 1, 1, 1, 1])
    assert held["C1"].tolist() == pytest.approx([0, 0, 0.5, 0.5, 1, 1, 1, 1])


def test_the_overlays_convert_days_to_the_panels_own_bars() -> None:
    """Three days on an hourly panel are 72 bars: the last part falls 72 bars after the first."""
    index = pd.date_range("2024-01-01", periods=24 * 8, freq="1h", tz="UTC")
    prices = pd.DataFrame({"C0": 100.0}, index=index)
    panel = MarketPanel(snapshot_id="hourly", prices=prices,
                        funding=pd.DataFrame(0.0, index=index, columns=["C0"]),
                        tradeable=pd.DataFrame(True, index=index, columns=["C0"]), meta={})
    target = pd.DataFrame({"C0": 0.0}, index=index)
    target.iloc[24:, 0] = 1.0
    held = GRADUAL_ENTRY.apply(target, Context(panel, None), {
        "tranches": 3, "over_days": 3, "only_if_profit": False})["C0"]
    assert held.iloc[24] == pytest.approx(1 / 3)
    assert held.iloc[24 + 36] == pytest.approx(2 / 3)
    assert held.iloc[24 + 72] == pytest.approx(1.0)
    assert held.iloc[24 + 71] < 1.0


def test_gradual_entry_leaves_exits_immediate_and_gradual_exit_leaves_entries_immediate() -> None:
    panel = flat_panel([100.0] * 12)
    target = pd.DataFrame({"C0": [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}, index=panel.prices.index,
                          dtype=float)
    entry = GRADUAL_ENTRY.apply(target, Context(panel, None), {
        "tranches": 2, "over_days": 2, "only_if_profit": False})["C0"]
    leave = GRADUAL_EXIT.apply(target, Context(panel, None), {
        "tranches": 2, "over_days": 2})["C0"]
    assert entry.iloc[6] == 0.0 and entry.iloc[1] == pytest.approx(0.5)
    assert leave.iloc[1] == 1.0 and leave.iloc[6] == pytest.approx(0.5)
    assert leave.iloc[8] == 0.0


def test_overlay_parameters_are_checked() -> None:
    for name, bad in [("gradual_entry", {"tranches": 1, "over_days": 3, "only_if_profit": False}),
                      ("gradual_exit", {"tranches": 2, "over_days": 0}),
                      ("own_results", {"window_days": 7, "floor": 2.0, "mode": "step",
                                       "up_days": 1}),
                      ("own_results", {"window_days": 7, "floor": 0.5, "mode": "wavy",
                                       "up_days": 1}),
                      ("vol_target", {"lookback_days": 30, "max_scale": 0.5}),
                      ("market_gate", {"detector": "direction_terciles", "trade_in": []})]:
        with pytest.raises(ValueError):
            parse_chain([{"name": name, "params": bad}])


def test_every_grid_is_a_family_not_one_value() -> None:
    for name in ("market_gate", "vol_target", "own_results", "gradual_entry", "gradual_exit"):
        overlay = get(name)
        assert len(overlay.grid()) >= 3
        assert overlay.coarse_grid() and len(overlay.coarse_grid()) <= len(overlay.grid())
        for point in overlay.grid().values():
            overlay.check(point)
    assert len(OWN_RESULTS.grid()) == 5 * 3 * 2 * 2     # windows x floors x modes x speeds
    assert {p["window_days"] for p in OWN_RESULTS.grid().values()} == {7, 14, 30, 60, 90}
    assert {p["floor"] for p in OWN_RESULTS.grid().values()} == {0.0, 0.25, 0.5}


def test_the_meta_gate_holds_only_on_the_days_the_second_level_says_earn(monkeypatch) -> None:
    """The stored level-2 decisions (`qlab regimes meta`) gate the book like any detector: a
    decision stamped at a close is known from that close on, and nothing is held before the
    first one."""
    import qlab.meta_label as meta_label

    panel = make_panel(rows=60, cols=3, snapshot="meta-gate")
    stamps = pd.date_range("2022-01-11", periods=50, freq="1D", tz="UTC")
    trade = pd.Series([k % 4 != 0 for k in range(50)], index=stamps)
    monkeypatch.setattr(meta_label, "load_decisions",
                        lambda name: pd.DataFrame({"trade": trade}))
    weights = Overlaid().target_weights(panel, overlaid(
        [{"name": "meta_gate", "params": {"name": "toy"}}], inner=f"{__name__}:Constant"))
    held = weights.abs().sum(axis=1) > 0
    assert not held.loc[:"2022-01-10"].any()            # no decision known yet
    assert held.loc["2022-01-12"] and not held.loc["2022-01-11"]
    assert not held.loc["2022-01-15"]                   # a "cash" decision (k = 4)
    assert get("meta_gate").grid_for(["a", "b"]) == {"meta-a": {"name": "a"},
                                                      "meta-b": {"name": "b"}}
    assert get("meta_gate").grid() == {}                # offered only where declared


def test_the_inner_strategys_own_flat_fees_are_kept() -> None:
    class Fees:
        name = "fees"

        def target_weights(self, panel, params):
            return Constant().target_weights(panel, params)

        def extra_costs(self, panel, params):
            return pd.Series(0.001, index=panel.prices.index)

    import sys

    sys.modules[__name__].Fees = Fees
    panel = make_panel(rows=10)
    params = overlaid([], inner=f"{__name__}:Fees")
    costs = Overlaid().extra_costs(panel, params)
    assert costs is not None and (costs == 0.001).all()
    assert Overlaid().extra_costs(panel, overlaid([])) is None
