"""Why the two venues differ (qlab.venue_gap): the numbers are worked by hand."""

from __future__ import annotations

from qlab import venue_gap as vg
from qlab.venue_gap import FailedRule, VenueFacts


def _binance(**kw) -> VenueFacts:
    base = dict(venue="binance", title="Binance", outcome="passed", route="needs-infrastructure",
                data_start="2020-01-01", data_end="2026-10-29", trial_id=10,
                taker_fee_bps=5.0, slippage_bps=0.9, money_outcome="failed",
                money_detail="нужно $1,000, своих $500: не помещается")
    base.update(kw)
    return VenueFacts(**base)


def _hl(**kw) -> VenueFacts:
    base = dict(venue="hyperliquid", title="Hyperliquid", outcome="failed", route="reject",
                data_start="2025-01-01", data_end="2026-10-29", trial_id=20,
                taker_fee_bps=4.5, slippage_bps=0.9, money_outcome="passed",
                money_detail="нужно $200 против денег стратегии $500")
    base.update(kw)
    return VenueFacts(**base)


# ---- words -------------------------------------------------------------------------------


def test_percent_has_a_decimal_comma_and_a_true_minus() -> None:
    assert vg.pct(-0.031) == "−3,1%"
    assert vg.pct(0.447, 0) == "45%"
    assert vg.pct(None) == "—"


def test_a_window_is_counted_in_days_by_hand() -> None:
    # 2025-01-01 .. 2026-10-29: 365 days of 2025 + 301 days of 2026 up to 10-29 = 666
    assert vg.window_days(_hl()) == 666
    # 2020-01-01 .. 2026-01-01 is 6 * 365 + 2 leap days (2020, 2024) = 2192; + 301 = 2493
    assert vg.window_days(_binance()) == 2493
    assert vg.window_days(_hl(data_start=None)) is None


def test_the_failed_rule_comes_with_its_numbers() -> None:
    rule = FailedRule("net_edge_positive", "cap_ann_return", -0.031, ">", 0.0)
    text = vg.rule_text(rule, _hl())
    assert text == "правило `net_edge_positive`: −3,1%, нужно больше 0,0%"


def test_the_noise_bar_failure_says_how_strong_the_bar_was() -> None:
    rule = FailedRule("shape_aware_edge", "noise_cap_return_percentile", 0.87, ">=", 0.99)
    weak = vg.rule_text(rule, _hl(noise_power=0.26))
    assert "87,0%" in weak and "не меньше 99,0%" in weak
    assert "сила планки 26%" in weak and "меньше 80%" in weak and "рано судить" in weak
    strong = vg.rule_text(rule, _hl(noise_power=0.9))
    assert "сила планки 90% — планка здесь различает" in strong
    unknown = vg.rule_text(rule, _hl(noise_power=None))
    assert "сила планки шума не измерена" in unknown


# ---- the profit made on coins the other venue does not list ----------------------------------


def test_profit_share_on_unlisted_coins_by_hand() -> None:
    net = {"BTC": 0.30, "DOGE": 0.10, "XYZ": 0.60, "ABC": -0.20}
    listed = {"BTC", "DOGE", "ABC"}
    share = vg.profit_share(net, lambda coin: coin in listed, venue="binance",
                            other="hyperliquid")
    # total 0.30 + 0.10 + 0.60 - 0.20 = 0.80; the one unlisted coin earned 0.60: 0.60 / 0.80
    assert abs(share.total - 0.80) < 1e-12
    assert abs(share.unlisted - 0.60) < 1e-12
    assert abs(share.share - 0.75) < 1e-12
    assert share.top == (("XYZ", 0.60),)
    assert (share.coins_unlisted, share.coins_held) == (1, 4)


def test_a_share_of_a_losing_window_is_not_defined() -> None:
    share = vg.profit_share({"A": -0.1, "B": 0.05}, lambda coin: coin == "B", venue="hyperliquid",
                            other="binance")
    assert share.share is None and share.why_not
    assert "не положительна" in vg.share_text(share)


def test_share_text_names_the_direction() -> None:
    share = vg.profit_share({"BTC": 0.2, "XYZ": 0.6}, lambda coin: coin == "BTC",
                            venue="binance", other="hyperliquid", window_from="2025-01-01",
                            window_to="2026-10-29")
    text = vg.share_text(share)
    assert text.startswith("75% прибыли Binance за 2025-01-01 — 2026-10-29")
    assert "с монет, которых нет на Hyperliquid (1 из 2 монет книги)" in text
    assert "XYZ 60,00%" in text


# ---- the reason as a whole -------------------------------------------------------------------


def test_one_venue_passed_and_the_other_did_not() -> None:
    rule = FailedRule("net_edge_positive", "cap_ann_return", -0.031, ">", 0.0)
    other = _hl(failed_rules=(rule,))
    pair = vg.applies(_binance(), other)
    assert pair is not None and pair[0].venue == "binance" and pair[1].venue == "hyperliquid"
    assert vg.applies(_binance(), _binance(venue="hyperliquid")) is None  # both passed
    assert vg.applies(other, _hl(venue="binance", title="Binance")) is None  # neither

    shares = [
        vg.profit_share({"BTC": 0.2, "XYZ": 0.6}, lambda c: c == "BTC", venue="binance",
                        other="hyperliquid"),
        vg.profit_share({"BTC": 0.1, "HYPE": -0.02}, lambda c: True, venue="hyperliquid",
                        other="binance"),
    ]
    gap = vg.compose(*pair, shares=shares)
    assert gap.rule_ru == "не сошлось: правило `net_edge_positive`: −3,1%, нужно больше 0,0%"
    kinds = [c.kind for c in gap.causes]
    assert kinds == ["unlisted_coins", "unlisted_coins", "windows", "costs", "money"]
    by_kind = {c.kind: c for c in gap.causes}
    costs = by_kind["costs"].text_ru
    assert "5 + 0,9 б.п." in costs and "4,5 + 0,9 б.п." in costs
    assert "(2 493 дн.)" in by_kind["windows"].text_ru and "(666 дн.)" in by_kind["windows"].text_ru
    assert "Binance: нужно $1,000, своих $500" in by_kind["money"].text_ru
    assert gap.text_ru.startswith("Hyperliquid не прошла, хотя Binance прошла: ")
    assert gap.short_ru == ("Hyperliquid: `net_edge_positive` −3,1%; "
                            "75% прибыли Binance — с монет без второй площадки")


def test_a_cause_that_cannot_be_measured_is_named_not_left_out() -> None:
    gap = vg.compose(_binance(), _hl(failed_rules=(
        FailedRule("net_edge_positive", "cap_ann_return", -0.01, ">", 0.0),)), shares=[],
        shares_why="расчёт ещё не сделан")
    first = gap.causes[0]
    assert first.kind == "unlisted_coins" and not first.measured
    assert "причина не измерена: расчёт ещё не сделан" in first.text_ru
    assert "причина не измерена" in gap.text_ru


def test_a_venue_with_no_run_gives_the_named_reason() -> None:
    other = _hl(outcome="not_run", route=None, trial_id=None, data_start=None, data_end=None,
                taker_fee_bps=None, slippage_bps=None, money_outcome="not_run", money_detail=None,
                not_run_reason="идентификатор второго этапа занят другой идеей")
    gap = vg.compose(_binance(), other, shares=[], shares_why="одна из площадок не проводилась")
    kinds = [c.kind for c in gap.causes]
    assert "no_run" in kinds and gap.rule_ru is None
    assert gap.text_ru.startswith("Hyperliquid не проводилась, хотя Binance прошла: ")
    assert ("на Hyperliquid прогона нет: идентификатор второго этапа занят другой идеей"
            in gap.text_ru)
    assert gap.short_ru.startswith("Hyperliquid: не проводилась (идентификатор")


def test_an_undecided_venue_names_the_rule_that_keeps_it_waiting() -> None:
    rule = FailedRule("shape_aware_edge", "noise_cap_return_percentile", 0.985, ">=", 0.99)
    other = _hl(outcome="undecided", route="needs-forward", noise_power=0.03,
                failed_rules=(rule,))
    gap = vg.compose(_binance(), other)
    assert gap.rule_ru.startswith("не хватило: правило `shape_aware_edge`: 98,5%")
    assert "не решила, хотя Binance прошла" in gap.text_ru


def test_both_passed_but_not_on_equal_terms() -> None:
    """The money fits on Hyperliquid and not on Binance; the Hyperliquid pass rests on the history
    check alone, and its latest year lost money."""
    a = _binance()
    b = _hl(outcome="passed", route="paper", consistency_trial=172434, noise_power=0.02,
            years=((2025, 0.12, 365), (2026, -0.153, 301)))
    assert vg.applies(a, b) is None and vg.uneven(a, b)
    gap = vg.compose(a, b, kind="uneven")
    assert gap.kind == "uneven" and gap.rule_ru is None
    assert [c.kind for c in gap.causes][:3] == ["money", "history_only", "recent_year"]
    assert gap.text_ru.startswith("обе площадки прошли правила, но не на равных: ")
    assert "только на сверке с историей Binance (прогон 172434)" in gap.text_ru
    assert "последний год 2026 (301 дн.) −15,3% от денег стратегии" in gap.text_ru
    assert gap.short_ru == ("деньги: Binance нужно $1,000, своих $500: не помещается; "
                            "Hyperliquid — только сверкой с историей (сила планки 2%); "
                            "Hyperliquid 2026 −15,3%")
    # two equal passes are not uneven
    assert not vg.uneven(_binance(money_outcome="passed"), _hl(outcome="passed"))


# ---- the stored measurement ------------------------------------------------------------------


def test_the_stored_measurement_belongs_to_two_trials(tmp_path) -> None:
    share = vg.profit_share({"BTC": 0.2, "XYZ": 0.6}, lambda c: c == "BTC", venue="binance",
                            other="hyperliquid")
    assert vg.needs_measure("idea", (10, 20), tmp_path)
    vg.save_stored("idea", (10, 20), [share], None, tmp_path)
    assert not vg.needs_measure("idea", (10, 20), tmp_path)
    found, why = vg.load_stored("idea", (10, 20), tmp_path)
    assert why is None and found[0].share == share.share and found[0].top == share.top
    # a newer run of either venue makes the number stale
    assert vg.needs_measure("idea", (10, 21), tmp_path)
    assert vg.load_stored("idea", (10, 21), tmp_path)[0] == []
    # nothing stored at all
    assert vg.load_stored("other", (1, 2), tmp_path)[1].startswith("расчёт ещё не сделан")
