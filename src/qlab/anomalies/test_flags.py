"""Each anomaly flag on synthetic series: it fires at its line and not below it, and says why
when it is not defined. The lines here are the test's own, not `night/anomalies.yaml`'s."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.anomalies import flags as fl
from qlab.anomalies.config import FLAGS, load_config, parse_config

RAW = {
    "version": "test.1", "enabled": True,
    "scan": {"routes": ["paper"], "latest_only": True, "max_scans_per_night": 3},
    "flags": {
        "implausible_day": {"single_move": 0.06, "market_move": 0.05, "sigmas": 8,
                            "rationale": "r"},
        "concentrated_instruments": {"min_instruments": 5, "top1_share": 0.25,
                                     "top3_share": 0.5, "rationale": "r"},
        "concentrated_coin_days": {"top_fraction": 0.1, "min_top": 2, "min_total": 0.05,
                                   "share": 0.5, "rationale": "r"},
        "tail_move_coin_days": {"min_move": 0.5, "share": 0.4, "rationale": "r"},
        "unlisted_on_execution_venue": {"share": 0.3, "rationale": "r"},
        "frozen_price": {"min_days": 3, "share": 0.1, "rationale": "r"},
        "untradeable_coin_days": {"min_volume_usd": 1000000, "delist_days": 7, "share": 0.2,
                                  "rationale": "r"},
        "venue_divergence": {"min_common_days": 10, "gap_per_year": 0.5, "rationale": "r"},
        "stand_vs_paper": {"gap": 0.05, "rationale": "r"},
        "too_good_for_family": {"min_members": 4, "fence_iqr": 3.0, "rationale": "r"},
    },
    "investigator": {"model": "opus", "max_turns": 10, "max_tokens_per_investigation": 100000,
                     "max_per_night": 2, "max_attempts": 2, "max_programmer_requests": 1,
                     "calls_dir": "calls", "passed_routes": ["paper"]},
}
CFG = parse_config(RAW)


def line(name: str):
    return CFG.flag(name)


def days(values, start="2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"),
                     dtype=float)


# --- the file -----------------------------------------------------------------------------


def test_every_flag_has_a_line_with_a_rationale_in_the_real_file() -> None:
    cfg = load_config()
    assert cfg is not None
    assert set(cfg.flags) == set(FLAGS)
    assert all(f.rationale for f in cfg.flags.values())


def test_a_missing_key_is_an_error_not_a_default() -> None:
    broken = {**RAW, "flags": {**RAW["flags"], "venue_divergence": {"rationale": "r"}}}
    with pytest.raises(ValueError, match="venue_divergence"):
        parse_config(broken)
    with pytest.raises(ValueError, match="investigator"):
        parse_config({k: v for k, v in RAW.items() if k != "investigator"})


# --- 1. implausible day -------------------------------------------------------------------


def _day_args(ret: float, n: float, gross: float, net: float = 0.0):
    return (days([0.001, ret, -0.002]), days([n] * 3), days([gross] * 3), days([net] * 3))


def test_a_day_beyond_what_n_positions_can_give_fires() -> None:
    # 20 dollar-neutral positions, gross 1: sd = 0.06/sqrt(20) = 1.34%, line = 8 sd = 10.7%
    quiet = fl.implausible_day(*_day_args(0.10, 20, 1.0), line("implausible_day"))
    fires = fl.implausible_day(*_day_args(0.12, 20, 1.0), line("implausible_day"))

    assert quiet.state == fl.QUIET and quiet.value == pytest.approx(0.10 / 0.1073, abs=0.01)
    assert fires.state == fl.FIRES and fires.value > 1.0
    assert fires.detail["top"][0]["day"] == "2026-01-02" and fires.detail["days_over"] == 1


def test_the_line_moves_with_the_number_of_positions() -> None:
    # The same +12% day: ordinary for a 2-position book, impossible for a 50-position one.
    few = fl.implausible_day(*_day_args(0.12, 2, 1.0), line("implausible_day"))
    many = fl.implausible_day(*_day_args(0.12, 50, 1.0), line("implausible_day"))

    assert few.state == fl.QUIET and many.state == fl.FIRES


def test_net_market_exposure_widens_the_line() -> None:
    # A long-only book (net = gross) carries the market's own move; a neutral one does not.
    neutral = fl.implausible_day(*_day_args(0.12, 20, 1.0, net=0.0), line("implausible_day"))
    long_only = fl.implausible_day(*_day_args(0.12, 20, 1.0, net=1.0), line("implausible_day"))

    assert neutral.state == fl.FIRES and long_only.state == fl.QUIET


def test_a_book_that_held_nothing_has_no_implausible_day() -> None:
    out = fl.implausible_day(days([0.0, 0.0]), days([0, 0]), days([0, 0]), days([0, 0]),
                             line("implausible_day"))

    assert out.state == fl.UNDEFINED and "нет дней с позициями" in out.reason


# --- 2. concentrated instruments ----------------------------------------------------------


def test_profit_on_one_or_three_coins_fires_at_either_line() -> None:
    spread = pd.Series([1.0] * 10)  # 10 coins, 10% each
    one = pd.Series([4.0] + [1.0] * 9)  # top1 = 4/13 = 31% >= 25%
    three = pd.Series([1.6, 1.6, 1.6] + [1.0] * 7)  # top1 = 16%, top3 = 4.8/11.8 = 41% < 50%
    three_heavy = pd.Series([2.0, 2.0, 2.0] + [1.0] * 7)  # top3 = 6/13 = 46%
    heavier = pd.Series([3.0, 3.0, 3.0] + [1.0] * 7)  # top3 = 9/16 = 56% >= 50%
    ln = line("concentrated_instruments")

    assert fl.concentrated_instruments(spread, ln).state == fl.QUIET
    assert fl.concentrated_instruments(one, ln).fired
    assert fl.concentrated_instruments(three, ln).state == fl.QUIET
    assert fl.concentrated_instruments(three_heavy, ln).state == fl.QUIET
    top3 = fl.concentrated_instruments(heavier, ln)
    assert top3.fired and top3.detail["top3_share"] == pytest.approx(9 / 16)


def test_a_short_list_or_a_loss_is_not_defined() -> None:
    ln = line("concentrated_instruments")

    assert fl.concentrated_instruments(pd.Series([5.0, 1.0, 1.0]), ln).state == fl.UNDEFINED
    assert "меньше" in fl.concentrated_instruments(pd.Series([5.0, 1.0, 1.0]), ln).reason
    lost = fl.concentrated_instruments(pd.Series([-5.0] + [1.0] * 5), ln)
    assert lost.state == fl.UNDEFINED and "не положительна" in lost.reason


# --- 3. concentrated coin-days ------------------------------------------------------------


def _coin_days(values) -> pd.Series:
    index = pd.MultiIndex.from_product([pd.date_range("2026-01-01", periods=len(values)), ["A"]])
    return pd.Series(values, index=index, dtype=float)


def test_profit_on_a_thousandth_of_the_days_fires() -> None:
    # 100 held coin-days, top_fraction 0.1 -> the best 10 days.
    lottery = _coin_days([10.0] * 10 + [0.1] * 90)  # best 10 = 100 of 109 = 92%
    even = _coin_days([1.0] * 100)  # best 10 = 10%
    ln = line("concentrated_coin_days")

    assert fl.concentrated_coin_days(lottery, 100, ln).fired
    assert fl.concentrated_coin_days(even, 100, ln).state == fl.QUIET
    out = fl.concentrated_coin_days(lottery, 100, ln)
    assert out.detail["top_k"] == 10 and out.value == pytest.approx(100 / 109)


def test_the_best_days_never_number_fewer_than_min_top() -> None:
    out = fl.concentrated_coin_days(_coin_days([5.0, 5.0, 1.0, 1.0]), 4,
                                    line("concentrated_coin_days"))

    assert out.detail["top_k"] == 2  # 10% of 4 is below min_top = 2


def test_coin_days_of_a_losing_or_a_tiny_run_are_not_defined() -> None:
    lost = fl.concentrated_coin_days(_coin_days([-1.0, 0.5]), 2, line("concentrated_coin_days"))
    tiny = fl.concentrated_coin_days(_coin_days([0.02, 0.01, 0.01]), 3,
                                     line("concentrated_coin_days"))

    assert lost.state == fl.UNDEFINED and "не положительна" in lost.reason
    assert tiny.state == fl.UNDEFINED and "меньше 5%" in tiny.reason


# --- 4. untradeable coin-days -------------------------------------------------------------


def _untradeable(union: float, *, total: float = 10.0):
    return fl.untradeable_coin_days(
        total=total, low_volume=union, near_delist=0.0, union=union, held_coin_days=100,
        low_volume_days=5, near_delist_days=0, unknown_volume_days=0,
        line=line("untradeable_coin_days"))


def test_profit_earned_where_nothing_could_be_traded_fires_at_its_share() -> None:
    assert _untradeable(1.9).state == fl.QUIET
    fired = _untradeable(2.0)
    assert fired.state == fl.FIRES and fired.value == pytest.approx(0.2)
    assert fired.detail["min_volume_usd"] == 1_000_000 and fired.detail["delist_days"] == 7


def test_untradeable_share_of_a_losing_run_is_not_defined() -> None:
    assert _untradeable(1.0, total=-1.0).state == fl.UNDEFINED


# --- 5. venue divergence ------------------------------------------------------------------


def test_two_venues_that_differ_on_the_common_days_fire() -> None:
    binance = days([0.002] * 20)  # about +107% a year
    hyper = days([0.0001] * 20, start="2026-01-01")  # about +4% a year
    close = days([0.00195] * 20)

    far = fl.venue_divergence(binance, hyper, line("venue_divergence"))
    near = fl.venue_divergence(binance, close, line("venue_divergence"))

    assert far.fired and far.value > 0.5 and far.detail["stage1_ann"] > far.detail["stage2_ann"]
    assert near.state == fl.QUIET and abs(near.value) < 0.5


def test_two_venues_of_a_book_that_does_not_reinvest_are_compared_by_sums() -> None:
    """On the strategy's own money (docs/CAPITAL_BASE.md) the totals are sums and the yearly
    figure is linear: 20 days of +0.2% a day are +4% in total, +73% a year, not the compounded
    +107%."""
    stage1, stage2 = days([0.002] * 20), days([0.002] * 20)
    out = fl.venue_divergence(stage1, stage2, line("venue_divergence"), additive=True)
    assert out.detail["stage1_total"] == pytest.approx(0.04)
    assert out.detail["stage1_ann"] == pytest.approx(0.04 * 365.25 / 20)
    assert out.state == fl.QUIET and out.value == pytest.approx(0.0)
    compounded = fl.venue_divergence(stage1, stage2, line("venue_divergence"))
    assert compounded.detail["stage1_total"] == pytest.approx(1.002 ** 20 - 1)


def test_only_the_common_days_are_compared_and_a_short_overlap_is_not_defined() -> None:
    long_run = days([0.01] * 100, start="2026-01-01")
    short_run = days([0.0] * 100, start="2026-04-05")  # overlaps only the last 6 days

    out = fl.venue_divergence(long_run, short_run, line("venue_divergence"))
    assert out.state == fl.UNDEFINED and "общих дней меньше" in out.reason

    longer = days([0.0] * 100, start="2026-03-15")  # 27 common days
    assert fl.venue_divergence(long_run, longer, line("venue_divergence")).detail["days"] == 27


# --- 6. stand against paper ---------------------------------------------------------------


def test_stand_and_paper_apart_by_the_gap_fire_only_on_one_window() -> None:
    ln = line("stand_vs_paper")

    fired = fl.stand_vs_paper(paper_return=0.01, stand_return=0.08, replay_return=None,
                              comparable=True, line=ln)
    quiet = fl.stand_vs_paper(paper_return=0.03, stand_return=0.06, replay_return=None,
                              comparable=True, line=ln)
    other_window = fl.stand_vs_paper(paper_return=0.01, stand_return=0.08, replay_return=None,
                                     comparable=False, line=ln)
    no_book = fl.stand_vs_paper(paper_return=None, stand_return=0.08, replay_return=None,
                                comparable=True, line=ln)

    assert fired.fired and fired.value == pytest.approx(0.07)
    assert quiet.state == fl.QUIET
    assert other_window.state == fl.UNDEFINED and "разные" in other_window.reason
    assert no_book.state == fl.UNDEFINED


# --- 7. too good for the family -----------------------------------------------------------


def test_a_return_beyond_the_far_out_fence_of_the_family_fires() -> None:
    family = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]  # Q1 0.1250, Q3 0.1750, fence 0.325
    ln = line("too_good_for_family")

    inside = fl.too_good_for_family(0.30, family, ln)
    outside = fl.too_good_for_family(0.40, family, ln)

    assert inside.state == fl.QUIET and outside.fired
    assert outside.line == pytest.approx(0.325) and outside.detail["members"] == 6


def test_a_small_family_has_no_spread_to_compare_with() -> None:
    out = fl.too_good_for_family(0.9, [0.1, 0.2], line("too_good_for_family"))

    assert out.state == fl.UNDEFINED and "меньше" in out.reason
    assert fl.too_good_for_family(None, [0.1] * 9, line("too_good_for_family")).state \
        == fl.UNDEFINED


def test_a_flag_round_trips_through_json() -> None:
    out = fl.implausible_day(*_day_args(0.12, 20, 1.0), line("implausible_day"))

    assert fl.FlagResult.from_json(out.to_json()) == out
    assert out.state_ru == "сработал" and out.title_ru


# --- the flags added after the XSMOM investigation ------------------------------------------


def test_profit_on_coins_that_jumped_fires_at_its_share() -> None:
    ln = line("tail_move_coin_days")

    def flag(tail_net: float):
        return fl.tail_move_coin_days(total=10.0, tail_net=tail_net, tail_days=3,
                                      held_coin_days=1000, top=[], line=ln)

    assert flag(3.9).state == fl.QUIET
    fired = flag(4.0)
    assert fired.fired and fired.value == pytest.approx(0.4)
    assert fired.detail["tail_day_share_of_held"] == pytest.approx(0.003)
    lost = fl.tail_move_coin_days(total=-1.0, tail_net=0.0, tail_days=0, held_coin_days=5,
                                  top=[], line=ln)
    assert lost.state == fl.UNDEFINED


def test_the_four_groups_of_the_xsmom_investigation_fall_on_the_right_side_of_the_line() -> None:
    """docs/research/xsmom-anomaly-2026-10-28.md: the shares of profit from coin-days with a
    move of 50% or more were 59% (Binance XSMOM), 53% (Hyperliquid XSMOM), 22% (trend) and 0%
    (Bv2); the line has to separate the first two from the last two."""
    ln = line("tail_move_coin_days")
    for share, fires in ((0.59, True), (0.53, True), (0.22, False), (0.0, False)):
        out = fl.tail_move_coin_days(total=1.0, tail_net=share, tail_days=1, held_coin_days=10,
                                     top=[], line=ln)
        assert out.fired is fires


def test_profit_on_coins_the_execution_venue_does_not_list_fires_at_its_share() -> None:
    ln = line("unlisted_on_execution_venue")

    def flag(unlisted: float, total: float = 10.0):
        return fl.unlisted_on_execution_venue(
            venue="hyperliquid", total=total, unlisted_net=unlisted, unlisted_days=4,
            held_coin_days=100, start="2025-01-01", top=[], line=ln)

    assert flag(2.9).state == fl.QUIET and flag(3.0).fired
    assert flag(3.0).detail["venue"] == "hyperliquid" and flag(3.0).detail["from"] == "2025-01-01"
    assert flag(1.0, total=0.0).state == fl.UNDEFINED


def test_a_price_that_does_not_move_for_days_fires_at_its_share() -> None:
    ln = line("frozen_price")

    quiet = fl.frozen_price(frozen_days=9, held_coin_days=100, longest=[], line=ln)
    fired = fl.frozen_price(frozen_days=10, held_coin_days=100,
                            longest=[{"instrument": "BTCST", "days": 243, "from": "2024-01-01"}],
                            line=ln)

    assert quiet.state == fl.QUIET and fired.fired
    assert fired.detail["longest"][0]["instrument"] == "BTCST"
    assert fl.frozen_price(frozen_days=0, held_coin_days=0, longest=[], line=ln).state \
        == fl.UNDEFINED
