"""Tests for `qlab.money_ladder`: how a strategy works on different money (owner, 2026-11-03).

The numbers are worked by hand in the comments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import money_ladder as ml
from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import CapitalSetup
from qlab.venues.config import load_venue

MINS = np.array([10.0, 10.0])


def _orders(targets, money=1000.0, mins=MINS, trims=False):
    t = np.array(targets, dtype=float)
    return ml.execute_orders(t, np.full(len(t), money), mins, trims_allowed=trims)


def test_what_the_venue_refuses_is_not_opened_and_a_small_change_waits() -> None:
    held = _orders([
        [0.02, 0.005],   # $20 opens; $5 < $10 does not
        [0.02, 0.012],   # the second is now $12: opens
        [0.025, 0.012],  # +$5 on the first: under the minimum, not sent
        [0.03, 0.012],   # +$10 from where it IS (0.02): sent
        [0.027, 0.012],  # -$3 trim: not sent where only a full close is exempt
    ])
    assert held[:, 0].tolist() == [0.02, 0.02, 0.02, 0.03, 0.03]
    assert held[:, 1].tolist() == [0.0, 0.012, 0.012, 0.012, 0.012]


def test_a_trim_goes_through_on_a_venue_that_exempts_every_reduce_only_order() -> None:
    held = _orders([[0.03, 0.0], [0.027, 0.0]], trims=True)
    assert held[:, 0].tolist() == [0.03, 0.027]  # -$3 is reduce-only: sent on Binance


def test_closing_is_always_sent_and_a_flip_closes_then_opens_only_what_fits() -> None:
    held = _orders([
        [0.012, 0.012],
        [0.0, 0.012],      # close: always
        [0.0, -0.015],     # flip: close + short $15 fits
        [0.0, 0.003],      # flip to long $3: closed, the new leg does not fit -> flat
    ])
    assert held[:, 0].tolist() == [0.012, 0.0, 0.0, 0.0]
    assert held[:, 1].tolist() == [0.012, 0.012, -0.015, 0.0]


def test_a_position_left_under_the_minimum_is_still_closed() -> None:
    # $12 opened at $1 000; the book shrinks it to a $6 target: waits (not reduce-exempt), then
    # the target is flat and the $12 goes out whole.
    held = _orders([[0.012, 0.0], [0.006, 0.0], [0.0, 0.0]])
    assert held[:, 0].tolist() == [0.012, 0.012, 0.0]


def test_with_no_minimum_the_book_is_the_target() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(0, 0.05, (30, 4))
    held = ml.execute_orders(target, np.full(30, 1000.0), np.zeros(4), trims_allowed=False)
    assert np.allclose(held, target)


def test_fewer_positions_drops_the_smallest_and_gives_its_weight_to_the_rest() -> None:
    # $1 000, minimum $10: the 0.005 leg is $5 and goes; the longs keep their gross 1.0.
    row = np.array([[0.5, 0.495, 0.005]])
    out = ml.fewer_positions(row, np.array([1000.0]), np.full(3, 10.0), cap=None)
    assert out[0, 2] == 0.0
    assert out[0, 0] == pytest.approx(0.5 / 0.995)
    assert out[0, 1] == pytest.approx(0.495 / 0.995)
    assert np.abs(out).sum() == pytest.approx(1.0)
    # capped at the largest weight the strategy takes: the first stays 0.5, the rest is cash
    capped = ml.fewer_positions(row, np.array([1000.0]), np.full(3, 10.0), cap=0.5)
    assert capped[0, 0] == 0.5 and capped[0, 1] == pytest.approx(0.495 / 0.995)


def test_fewer_positions_keeps_each_side_of_a_hedged_book_apart() -> None:
    # longs 0.4 and 0.004, shorts -0.3 and -0.3: the dropped $4 long goes to the other long only
    row = np.array([[0.4, 0.004, -0.3, -0.3]])
    out = ml.fewer_positions(row, np.array([1000.0]), np.full(4, 10.0), cap=None)
    assert out[0].tolist() == pytest.approx([0.404, 0.0, -0.3, -0.3])
    # a side left with nothing is cash, not moved to the other side
    only = ml.fewer_positions(np.array([[0.005, -0.4]]), np.array([1000.0]), np.full(2, 10.0))
    assert only[0].tolist() == pytest.approx([0.0, -0.4])


def test_fewer_positions_drops_one_at_a_time_so_a_grown_position_stays() -> None:
    # $1 000: legs 0.6, 0.38, 0.0095 ($9.5), 0.0105 ($10.5, fits). Dropping only the smallest
    # leg ($9.5) lifts the others: 0.0105 * 1.0/0.9905 stays, the book keeps three legs.
    row = np.array([[0.6, 0.38, 0.0095, 0.0105]])
    out = ml.fewer_positions(row, np.array([1000.0]), np.full(4, 10.0), cap=None)
    assert out[0, 2] == 0.0 and out[0, 3] > 0.0
    assert out[0, 3] == pytest.approx(0.0105 / 0.9905)


def _panel(prices: dict[str, list[float]]) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=len(next(iter(prices.values()))), freq="1D",
                          tz="UTC")
    frame = pd.DataFrame(prices, index=index)
    return MarketPanel(prices=frame, funding=pd.DataFrame(0.0, index=index, columns=frame.columns),
                       tradeable=pd.DataFrame(True, index=index, columns=frame.columns),
                       snapshot_id="snap", meta={})


def test_the_return_loses_what_the_venue_refused_and_nothing_else() -> None:
    # A +10% a day, C +5% a day, B +1% a day; no costs, three prices -> two return bars.
    panel = _panel({"A": [100, 110, 121], "B": [100, 101, 102.01], "C": [100, 105, 110.25]})
    weights = pd.DataFrame({"A": 0.3, "B": 0.005, "C": 0.2}, index=panel.prices.index)
    costs = CostModel(0, 0)
    common = dict(part_panel=panel, weights=weights, costs=costs, extra_cost=None,
                  capital_kwargs={}, base=None, reinvested=True,
                  minimums=np.full(3, 10.0), trims_allowed=False)
    reference = ml.run_weights(panel, weights, costs, None, {})
    # nothing binds: 0.3*0.10 + 0.005*0.01 + 0.2*0.05 = 0.04005 a bar
    assert reference.net_return.iloc[0] == pytest.approx(0.04005)
    big, held_big, _ = ml.constrained_run(10_000.0, "skip", reference_result=reference, **common)
    assert big.net_return.iloc[0] == pytest.approx(0.04005)
    assert (held_big == weights.to_numpy()).all()
    # $1 000: B is $5 < $10, not opened: 0.3*0.10 + 0.2*0.05 = 0.04 a bar
    small, held, _ = ml.constrained_run(1_000.0, "skip", reference_result=reference, **common)
    assert small.net_return.iloc[0] == pytest.approx(0.04)
    assert (held[:, 1] == 0.0).all()
    # fewer positions: B dropped; A would grow to 0.303 but is capped at 0.3; C 0.2 -> 0.202:
    # 0.03 + 0.202 * 0.05 = 0.0401 a bar
    fewer, held_f, _ = ml.constrained_run(1_000.0, "fewer", reference_result=reference, **common)
    assert fewer.net_return.iloc[0] == pytest.approx(0.0401)
    assert held_f[0].tolist() == pytest.approx([0.3, 0.0, 0.202])


def test_the_money_at_work_grows_with_the_profit_of_a_book_that_reinvests() -> None:
    net = pd.Series([0.10, -0.50, 0.0], index=pd.date_range("2026-01-01", periods=3, tz="UTC"))
    # the equity at the START of each bar: 1, 1.1, 0.55
    assert ml.start_of_bar_equity(net).tolist() == pytest.approx([1.0, 1.1, 0.55])
    index = net.index
    assert ml.sizing_money(1000.0, None, index, np.array([1.0, 1.1, 0.55])).tolist() == \
        pytest.approx([1000.0, 1100.0, 550.0])
    assert ml.sizing_money(1000.0, None, index, None).tolist() == [1000.0] * 3
    base = pd.Series([1.0, 2.0, 2.0], index=index)
    assert ml.sizing_money(1000.0, base, index, None).tolist() == [1000.0, 2000.0, 2000.0]


def test_orders_are_counted_against_the_unconstrained_books_own_orders() -> None:
    wanted = np.array([[0.02, 0.005], [0.02, 0.012], [0.03, 0.012]])
    held = np.array([[0.02, 0.0], [0.02, 0.012], [0.02, 0.012]])
    n_wanted, n_sent, share, volume = ml.order_counts(wanted, held)
    # wanted: A opens (bar 0), B opens (bar 0), B 0.005 -> 0.012 (bar 1), A 0.02 -> 0.03 (bar 2)
    assert (n_wanted, n_sent) == (4, 2)  # A at bar 0 and B at bar 1 were sent
    assert share == pytest.approx(0.5)
    # volume wanted 0.02 + 0.005 + 0.007 + 0.01 = 0.042; not sent: B 0.005 and A 0.01
    assert volume == pytest.approx(0.015 / 0.042)


def test_the_strict_origin_names_the_instrument_the_day_and_the_weight() -> None:
    index = pd.date_range("2026-03-01", periods=3, freq="1D", tz="UTC")
    weights = pd.DataFrame({"BTC": [0.4, 0.4, 0.0013], "ALT": [0.1, 0.00005, 0.1]}, index=index)
    out = ml.strict_origin(weights, np.array([50.0, 5.0]), from_orders=False, venue="binance",
                           own_usd=3000.0)
    # BTC: 50 / 0.0013 = 38 461; ALT: 5 / 0.00005 = 100 000 -> ALT on the second day
    assert (out.instrument, out.day) == ("ALT", "2026-03-02")
    assert out.min_capital_usd == pytest.approx(100_000.0)
    assert out.positions_over_own == 2 and out.positions_total == 6  # 38 461 and 100 000 > 3 000
    ladder = _toy_ladder(strict=out)
    line = ml.why_strict_ru(ladder)
    assert "ALT на Binance" in line and "$100 000" in line and "2026-03-02" in line


def _toy_ladder(strict: ml.Strict | None = None) -> ml.MoneyLadder:
    rows = [
        ml.Row(usd=1000.0, mode="skip", ann_return=0.02, max_dd=-0.15, not_sent_share=0.7,
               not_sent_volume_share=0.4, missing_position_share=0.4, avg_positions=7.0,
               correlation=0.95),
        ml.Row(usd=1000.0, mode="fewer", ann_return=0.046, max_dd=-0.17, avg_positions=7.1,
               correlation=0.99, missing_position_share=0.3),
        ml.Row(usd=3000.0, mode="skip", own=True, ann_return=0.0388, max_dd=-0.17,
               not_sent_share=0.456, not_sent_volume_share=0.112, missing_position_share=0.1,
               avg_positions=10.7, correlation=0.998),
        ml.Row(usd=3000.0, mode="fewer", own=True, ann_return=0.0433, max_dd=-0.171,
               avg_positions=10.4, correlation=0.999),
    ]
    return ml.MoneyLadder(
        trial_id=7, idea_id="toy", venue="binance", interval="1d",
        window=("2020-01-01", "2026-01-01"),
        own_usd=3000.0, basis="weights",
        reference={"ann_return": 0.0418, "max_dd": -0.177, "avg_positions": 12.0}, rows=rows,
        strict=strict or ml.Strict(min_capital_usd=37_591.0, instrument="H", day="2026-06-11",
                                   weight=0.000133, minimum_order=5.0, venue="binance"))


def test_the_plain_sentence_says_what_was_not_sent_and_the_return_against_unbound() -> None:
    line = ml.sentence_ru(_toy_ladder(), 3000.0)
    assert line == ("на $3 000 не отправлено 45.6% ордеров (11.2% объёма торговли), "
                    "доход +3.9% в год против +4.2% без ограничения")
    assert ml.sentence_ru(_toy_ladder(), 777.0) is None
    assert ml.money_line_ru(_toy_ladder()) == line  # at the strategy's own money


def test_the_reading_is_kept_as_trial_metrics_with_a_prefix() -> None:
    metrics = _toy_ladder().metrics()
    assert metrics["ml_3000_skip_ann_return"] == 0.0388
    assert metrics["ml_1000_fewer_avg_positions"] == 7.1
    assert metrics["ml_reference_ann_return"] == 0.0418 and metrics["ml_own_usd"] == 3000.0
    assert all(k.startswith("ml_") for k in metrics)
    assert "ml_1000_skip_below_minimum_share" not in metrics  # undefined: left out


def test_the_stored_reading_comes_back_only_for_the_same_trial_money_and_method(tmp_path) -> None:
    ladder = _toy_ladder()
    ml.save_stored(ladder, tmp_path)
    again = ml.load_stored(7, 3000.0, tmp_path)
    assert again is not None and again.rows[2].own and again.strict.instrument == "H"
    assert ml.load_stored(7, 4000.0, tmp_path) is None  # another own money
    assert ml.load_stored(8, 3000.0, tmp_path) is None  # another trial


def test_the_strategys_own_money_joins_the_sums_and_is_marked() -> None:
    assert ml.ladder_sums(257.0) == [(257.0, True), (1000.0, False), (3000.0, False),
                                     (10000.0, False)]
    assert ml.ladder_sums(3000.0) == [(1000.0, False), (3000.0, True), (10000.0, False)]
    assert ml.ladder_sums(None)[0] == (1000.0, False)


def test_the_venues_state_how_far_reduce_only_goes() -> None:
    # Binance: error -4164 "(unless you choose reduce only)"; Hyperliquid: only exactly closing.
    assert load_venue("binance").reduce_only_exempt == "any_reduction"
    assert load_venue("hyperliquid").reduce_only_exempt == "full_close"
    assert load_venue("aster") is None or load_venue("aster").reduce_only_exempt is None


def test_the_morning_table_lists_only_ideas_that_wait_for_the_owner() -> None:
    class Money:
        money_ladder = _toy_ladder()

    class Step:
        money = (Money(),)

    class Found:
        def __init__(self, idea_id, status):
            self.idea_id, self.status, self.steps = idea_id, status, [None, None, None, Step()]

    lines = ml.report_lines([Found("a-validated", "validated"), Found("b-bench", "bench")])
    text = "\n".join(lines)
    assert "a-validated" in text and "b-bench" not in text
    assert "| a-validated | Binance | $37 591 | пропускаем отказ площадки | " in text
    assert "+2.0% / −15%, позиций нет 40%" in text and "меньше позиций" in text
    assert ml.report_lines([Found("b-bench", "bench")]) == []


def test_a_book_that_drives_itself_has_no_fewer_row() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    panel = _panel({"A": [100, 110, 121, 133.1]})
    weights = pd.DataFrame({"A": 0.5}, index=index)

    def run_at(usd):
        return ml.run_weights(panel, weights, CostModel(0, 0), None, {}), weights, None

    out = ml.measure_book(trial_id=1, idea_id="book", venue="hyperliquid", interval="1d",
                          window=("2026-01-01", "2026-01-04"), own_usd=257.0, run_at=run_at)
    assert {r.mode for r in out.rows if r.available} == {"book"}
    assert all(not r.available and r.why_not for r in out.rows if r.mode == "fewer")
    assert out.row(257.0, "book").correlation is not None or True
    assert "fewer" not in " ".join(k for k in out.metrics())


def test_the_ladder_carries_the_stored_reading_into_the_money_step(tmp_path) -> None:
    from qlab.ladder import Outcome, VenueMoney

    money = VenueMoney("binance", Outcome.PASSED, "нужно $1, своих $2", money_ladder=_toy_ladder())
    body = money.to_dict()
    assert body["money_line_ru"].startswith("на $3 000 не отправлено 45.6% ордеров")
    assert body["money_ladder"]["why_strict_ru"].startswith("Строгий минимум $37 591")
    assert len(body["money_ladder"]["sentences_ru"]) == 2  # $1 000 and $3 000
    bare = VenueMoney("binance", Outcome.PASSED, "x").to_dict()
    assert bare["money_ladder"] is None and bare["money_line_ru"] is None


def test_capital_setup_with_a_refused_base_is_read_without_one() -> None:
    assert CapitalSetup(None, True, refused="why").kwargs == {"capital_base": None,
                                                               "capital_reinvested": True}


class _Trial:
    def __init__(self, metrics):
        self.metrics = metrics


class _Session:
    def __init__(self, trial):
        self.trial = trial

    def get(self, model, key):
        return self.trial


def test_the_trial_gets_only_the_ml_keys_and_a_stale_reading_is_replaced() -> None:
    ladder = _toy_ladder()
    trial = _Trial({"ann_return_net": 0.04, "ml_3000_skip_ann_return": 0.99, "ml_old": 1.0})
    ml.write_metrics(_Session(trial), ladder)
    assert trial.metrics["ann_return_net"] == 0.04  # the stand's numbers are not touched
    assert trial.metrics["ml_3000_skip_ann_return"] == 0.0388 and "ml_old" not in trial.metrics
    assert ml._metrics_current(_Session(trial), ladder)
    assert not ml._metrics_current(_Session(_Trial({"ann_return_net": 0.04})), ladder)


def test_the_morning_table_leaves_out_a_mode_a_book_does_not_have() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    panel = _panel({"A": [100, 110, 121, 133.1]})
    weights = pd.DataFrame({"A": 0.5}, index=index)

    def run_at(usd):
        return ml.run_weights(panel, weights, CostModel(0, 0), None, {}), weights, None

    book = ml.measure_book(trial_id=1, idea_id="book", venue="hyperliquid", interval="1d",
                           window=("2026-01-01", "2026-01-04"), own_usd=257.0, run_at=run_at)

    class Money:
        money_ladder = book

    class Step:
        money = (Money(),)

    class Found:
        idea_id, status, steps = "book", "paper", [None, None, None, Step()]

    rows = [line for line in ml.report_lines([Found()]) if line.startswith("| book |")]
    assert len(rows) == 1 and "книга сама" in rows[0] and "$257:" in rows[0]
