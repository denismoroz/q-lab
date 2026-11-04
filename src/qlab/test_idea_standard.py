"""The standard of a strategy page, checked by code (docs/IDEA_STANDARD.md): the fourteen items for
an idea with everything, a candidate, a rejected idea with a chart, a narrowed coin list and a
stage missing on Hyperliquid. The facts are built by hand: `check` is a pure function of them."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from qlab import idea_standard as std
from qlab.idea_standard import CoinFacts, Facts, StageFacts

BINANCE_START = date(2019, 9, 8)
HL_START = date(2025, 1, 1)
STARTS = {"binance": BINANCE_START, "hyperliquid": HL_START}


def _run(venue: str, *, trial: int, rules: str = "2026-11-04.1", capital: float | None = 8000.0):
    return {"idea_id": "x", "trial_id": trial, "route": "paper", "venue": venue,
            "data_start": "2020-01-01", "data_end": "2026-11-02", "rules_version": rules,
            "noise_percentile": 0.99, "noise_power": 0.9, "min_capital_usd": capital}


def _step(number: int, venue: str, *, trial: int | None = None, reason: str | None = None,
          rules: str = "2026-11-04.1") -> dict:
    run = None if trial is None else _run(venue, trial=trial, rules=rules)
    return {"number": number, "venue": venue, "run": run,
            "outcome": "passed" if run else "not_run",
            "outcome_ru": "пройден" if run else "не проводился",
            "reason": reason, "reason_ru": f"слова: {reason}" if reason else None, "money": []}


def _ladder(*, binance: dict, hyperliquid: dict, money_rows=(), money_reason=None) -> dict:
    steps = [{"number": 1, "venue": None, "run": None}, binance, hyperliquid,
             {"number": 4, "venue": None, "run": None, "money": list(money_rows),
              "reason_ru": money_reason},
             {"number": 5}, {"number": 6}, {"number": 7}]
    return {"steps": steps}


def _trial(*, on_capital=True, fit=True, judged_on_forward=1, years=True, noise=0.99,
           power=0.9) -> dict:
    headline = {"on_capital": on_capital, "ann_return": 0.05, "max_dd": -0.1, "sharpe": 0.8,
                "ann_return_ex_best_1pct": 0.02, "noise_percentile": noise,
                "fit": {"ann_return": 0.06} if fit else None}
    return {"route": "paper", "route_reason": "r", "headline": headline,
            "metrics": {"judged_on_forward": judged_on_forward, "noise_cap_power": power},
            "year_view": {"years": [{"year": 2025}], "capital": {"years": [{"year": 2025}]}}
            if years else None}


def _stage(number: int, venue: str, trial: int | None, *, window: str | None = "whole",
           start: date | None = None, coins: CoinFacts | None = None, **more) -> StageFacts:
    return StageFacts(
        number=number, idea_id=f"x{'' if number == 1 else '-hl'}", venue=venue, trial_id=trial,
        series_window=window, spec_start=start or STARTS[venue], spec_end=date(2026, 11, 2),
        run_end=date(2026, 11, 2), has_spec=True,
        coins=coins or CoinFacts("code", "монеты выбирает правило кода"),
        costs={"taker_fee_bps": 5.0, "slippage_bps": 0.9}, **more)


def _full(**changes) -> Facts:
    money = [{"venue": "binance", "money_line_ru": "на $3 000 …"},
             {"venue": "hyperliquid", "money_line_ru": "на $3 000 …"}]
    base = dict(
        idea_id="x", status="validated", has_run=True, plain_source="family", card="x",
        outcome={"trial": _trial()}, rules_version="2026-11-04.1",
        ladder=_ladder(binance=_step(2, "binance", trial=1), hyperliquid=_step(3, "hyperliquid",
                                                                              trial=2),
                       money_rows=money),
        stages=(_stage(1, "binance", 1), _stage(2, "hyperliquid", 2)), venue_starts=STARTS,
        double_checked=True)
    base.update(changes)
    return Facts(**base)


def _states(items) -> dict[str, str]:
    return {i.key: i.state for i in items}


def test_there_are_fourteen_items_in_the_order_of_the_standard_doc() -> None:
    assert [d.number for d in std.ITEMS] == list(range(1, 15))
    assert [d.key for d in std.ITEMS][:3] == ["plain", "card", "outcome"]
    doc = Path("docs/IDEA_STANDARD.md").read_text(encoding="utf-8")
    section = doc.split("## Что должно быть у каждой стратегии")[1].split("##")[0]
    assert section.count("\n1. ") == 1 and "\n14. **Проверка на двойника**" in section


def test_an_idea_with_everything_has_all_fourteen() -> None:
    items = std.check(_full())

    assert len(items) == 14
    assert all(i.state == "есть" for i in items), [(i.number, i.reason) for i in items
                                                  if i.state != "есть"]
    assert std.have_count(items) == 14
    data = next(i for i in items if i.key == "data")
    assert "Binance: с 2019-09-08 по 2026-11-02" in data.detail
    assert "с начала данных площадки" in data.detail


def test_a_candidate_gets_not_needed_with_the_reason_for_what_needs_a_run() -> None:
    facts = Facts(idea_id="c", status="candidate", has_run=False, plain_source="card", card="c",
                  ladder=_ladder(binance=_step(2, "binance", reason="card_without_implementation"),
                                 hyperliquid=_step(3, "hyperliquid",
                                                   reason="card_without_implementation")))
    items = std.check(facts)
    states = _states(items)

    for key in ("outcome", "chart", "years", "numbers", "noise", "money", "venues", "data",
                "coins", "costs", "double"):
        assert states[key] == "не нужно", key
    chart = next(i for i in items if i.key == "chart")
    assert chart.reason == "ещё не прогонялась"
    assert states["plain"] == states["card"] == states["ladder"] == "есть"
    assert "нет" not in states.values()


def test_a_rejected_idea_with_a_stored_series_still_has_the_chart() -> None:
    rejected = _trial()
    rejected["route"] = "reject"
    facts = _full(status="rejected", outcome={"trial": rejected})

    states = _states(std.check(facts))

    assert states["chart"] == states["outcome"] == "есть"


@pytest.mark.parametrize("window,word", [
    (None, "ряд денег ещё не построен — будет после пересчёта"),
    ("forward_only", "сохранён только отрезок проверки вперёд"),
])
def test_the_chart_is_missing_unless_every_stage_has_a_whole_window_series(window, word) -> None:
    facts = _full(stages=(_stage(1, "binance", 1), _stage(2, "hyperliquid", 2, window=window)))

    chart = next(i for i in std.check(facts) if i.key == "chart")

    assert chart.state == "нет" and word in chart.reason and "Hyperliquid" in chart.reason


def test_a_coin_list_chosen_by_an_agent_without_a_source_is_a_defect() -> None:
    narrowed = CoinFacts("agent", "список монет сужен без источника: у родителя 4 монеты, а в "
                                  "этом опыте 2", ("BTC", "ETH"))
    facts = _full(stages=(_stage(1, "binance", 1, coins=narrowed), _stage(2, "hyperliquid", 2)))

    coins = next(i for i in std.check(facts) if i.key == "coins")

    assert coins.state == "нет" and coins.reason.startswith("список монет сужен без источника")


def test_coin_sources_are_listed_when_they_are_given() -> None:
    owner = CoinFacts("owner", "BTC, ETH — слово владельца")
    facts = _full(stages=(_stage(1, "binance", 1, coins=owner), _stage(2, "hyperliquid", 2)))

    coins = next(i for i in std.check(facts) if i.key == "coins")

    assert coins.state == "есть" and "слово владельца" in coins.detail
    assert "правило кода" in coins.detail


def test_a_stage_not_run_on_hyperliquid_is_a_gap_named_in_words() -> None:
    facts = _full(
        ladder=_ladder(binance=_step(2, "binance", trial=1),
                       hyperliquid=_step(3, "hyperliquid", reason="waits_for_night")),
        stages=(_stage(1, "binance", 1),))

    venues = next(i for i in std.check(facts) if i.key == "venues")

    assert venues.state == "нет" and "на Hyperliquid ещё не прогонялась" in venues.reason
    assert "слова: waits_for_night" in venues.reason


def test_a_venue_without_a_version_by_nature_is_answered_by_its_reason() -> None:
    facts = _full(
        ladder=_ladder(binance=_step(2, "binance", reason="no_version_on_venue"),
                       hyperliquid=_step(3, "hyperliquid", trial=2),
                       money_rows=[{"venue": "hyperliquid", "money_line_ru": "x"}]),
        stages=(_stage(2, "hyperliquid", 2),))

    venues = next(i for i in std.check(facts) if i.key == "venues")

    assert venues.state == "есть" and "слова: no_version_on_venue" in venues.detail


def test_a_run_under_old_rules_has_no_last_third_split() -> None:
    old = _full(rules_version="2026-11-03.1")

    numbers = next(i for i in std.check(old) if i.key == "numbers")

    assert numbers.state == "нет" and "старым правилам" in numbers.reason
    assert "2026-11-04.1" in numbers.reason


def test_numbers_not_on_the_strategys_money_are_missing() -> None:
    facts = _full(outcome={"trial": _trial(on_capital=False)})

    numbers = next(i for i in std.check(facts) if i.key == "numbers")

    assert numbers.state == "нет" and "не на деньгах стратегии" in numbers.reason


def test_a_noise_bar_without_power_says_it() -> None:
    facts = _full(outcome={"trial": _trial(power=None)})
    facts.outcome["trial"]["metrics"].pop("noise_cap_power")

    noise = next(i for i in std.check(facts) if i.key == "noise")

    assert noise.state == "нет" and "хватает ли данных" in noise.reason


def test_money_on_the_ladder_must_be_measured_for_every_venue_with_a_run() -> None:
    facts = _full(ladder=_ladder(
        binance=_step(2, "binance", trial=1), hyperliquid=_step(3, "hyperliquid", trial=2),
        money_rows=[{"venue": "binance", "money_line_ru": "x"}]))

    money = next(i for i in std.check(facts) if i.key == "money")

    assert money.state == "нет" and "Hyperliquid" in money.reason and "$1k" in money.reason


def test_money_under_rules_2026_11_04_2_needs_the_own_money_book_measured() -> None:
    """Rules 2026-11-04.2: step 4 judges the «fewer positions» book on the strategy's own money;
    a venue where it was not measured is a gap, named with the reason."""
    unmeasured = {"venue": "hyperliquid", "money_line_ru": "x", "rule": "fewer_own",
                  "outcome": "undecided",
                  "detail": "на своих $3 000 при меньшем числе позиций не измерено: расчёт упал"}
    facts = _full(ladder=_ladder(
        binance=_step(2, "binance", trial=1), hyperliquid=_step(3, "hyperliquid", trial=2),
        money_rows=[{"venue": "binance", "money_line_ru": "x", "rule": "fewer_own",
                     "outcome": "passed"}, unmeasured]))

    money = next(i for i in std.check(facts) if i.key == "money")

    assert money.state == "нет" and "не измерено: расчёт упал" in money.reason
    assert "Binance" not in money.reason


# --- the first day of the data ---------------------------------------------------------------


def _start(**kw) -> tuple[bool, str]:
    stage = _stage(1, "binance", 1, **kw)
    return std.explain_start(stage, BINANCE_START)


def test_a_start_at_the_venues_first_day_needs_no_reason() -> None:
    ok, words = _start()
    assert ok and words == "с начала данных площадки"


def test_a_later_start_is_explained_by_warm_up_when_the_strategy_needs_that_history() -> None:
    ok, words = _start(start=date(2020, 4, 1), warmup_days=200)

    assert ok and "разогрев" in words and "200" in words


def test_a_later_start_is_explained_by_the_listing_day_of_a_coin() -> None:
    ok, words = _start(start=date(2020, 9, 25), coverage_day=date(2020, 9, 23),
                       coverage_coin="AVAX")

    assert ok and "AVAX" in words and "2020-09-23" in words


def test_a_later_start_is_explained_by_the_knowable_coins_check() -> None:
    ok, words = _start(start=date(2023, 1, 10), knowable_from=date(2023, 1, 5))

    assert ok and "очевидным выбором" in words


def test_a_later_start_with_a_reason_written_in_the_spec_is_accepted() -> None:
    ok, words = _start(start=date(2023, 4, 28), data_reason="спот Binance с этого дня")

    assert ok and "спот Binance" in words


def test_a_start_that_a_person_picked_is_a_shorter_period_than_available() -> None:
    day = date(2025, 6, 1)
    ok, words = _start(start=day, judged_from=day, instruments_chosen_at=day)

    assert not ok
    assert words.startswith("период короче доступного: начинается с 2025-06-01, потому что")
    assert "человек" in words and "2019-09-08" in words


def test_the_data_item_flags_a_later_start_without_a_known_reason() -> None:
    facts = _full(stages=(_stage(1, "binance", 1, start=date(2023, 4, 28)),
                          _stage(2, "hyperliquid", 2)))

    data = next(i for i in std.check(facts) if i.key == "data")

    assert data.state == "нет" and "период короче доступного" in data.reason
    assert "Binance: с 2023-04-28" in data.reason


def test_warm_up_is_read_off_the_longest_lookback_of_the_parameters() -> None:
    params = {"lookbacks": [20, 60, 120], "vol_window": 30, "nested": {"min_history_days": 200},
              "coins": ["BTC"], "flag": True}

    assert std.warmup_days(params) == 200
    assert std.warmup_days({"window_bars": 48}, "1h") == 2
    assert std.warmup_days({"coins": ["BTC"]}) is None


# --- the coin list of a real spec ------------------------------------------------------------


def _spec_yaml(idea_id: str, coins: list[str], **more) -> dict:
    return {"idea_id": idea_id, "title": idea_id, "code_ref": "qlab.strategies.live.bv2:LiveBv2",
            "params": {"coins": coins},
            "data": {"source": "binance", "interval": "1d", "start": "2024-01-01",
                     "end": "2026-01-01"},
            "costs": {"taker_fee_bps": 5.0, "slippage_bps": 0.9}, "min_leg_notional": 5.0,
            "unexpressed_mechanisms": [], **more}


def _root_with_parent(tmp_path: Path) -> Path:
    (tmp_path / "night").mkdir()
    (tmp_path / "specs").mkdir()
    (tmp_path / "night" / "coin_scope.yaml").write_text(yaml.safe_dump({"families": {
        "bvx": {"parent": "specs/bvx-live.yaml", "prefixes": ["bvx-"]}}}))
    (tmp_path / "specs" / "bvx-live.yaml").write_text(yaml.safe_dump(
        _spec_yaml("bvx-live", ["BTC", "ETH", "SOL", "AVAX"])))
    return tmp_path


def _load(tmp_path: Path, name: str, payload: dict):
    from qlab.pipeline.spec import load_spec

    path = tmp_path / "specs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_spec(path)


def test_a_narrowed_coin_list_without_a_source_is_found_in_a_real_spec(tmp_path) -> None:
    root = _root_with_parent(tmp_path)
    spec = _load(root, "bvx-two", _spec_yaml("bvx-two", ["BTC", "ETH"]))

    found = std.coin_facts(spec, status="bench", view=None, knowable_plain=None, root=root)

    assert found.kind == "agent" and found.defect
    assert found.text.startswith("список монет сужен без источника")


def test_a_narrowed_list_with_the_owners_word_is_not_a_defect(tmp_path) -> None:
    root = _root_with_parent(tmp_path)
    spec = _load(root, "bvx-two", _spec_yaml(
        "bvx-two", ["BTC", "ETH"],
        sources={"coins": "владелец, 2026-11-04: «только BTC и ETH, остальное шумит»"}))

    found = std.coin_facts(spec, status="bench", view=None, knowable_plain=None, root=root)

    assert found.kind == "owner" and not found.defect


def test_the_production_list_of_the_family_is_a_source(tmp_path) -> None:
    root = _root_with_parent(tmp_path)
    spec = _load(root, "bvx-same", _spec_yaml("bvx-same", ["BTC", "ETH", "SOL", "AVAX"]))

    found = std.coin_facts(spec, status="bench", view=None, knowable_plain=None, root=root)

    assert found.kind == "production" and "AVAX" in found.text


def test_a_list_nobody_gave_a_source_for_is_chosen_by_an_agent(tmp_path) -> None:
    root = _root_with_parent(tmp_path)
    spec = _load(root, "lone", _spec_yaml("lone", ["BTC", "ETH"]))

    found = std.coin_facts(spec, status="bench", view=None, knowable_plain=None, root=root)

    assert found.kind == "agent" and "выбраны агентом без источника" in found.text


# --- totals, text, the morning line ----------------------------------------------------------


def _row(idea_id: str, items) -> dict:
    return {"id": idea_id, "title": idea_id, "status": "bench", "family": "bv2",
            "have": std.have_count(items), "items": std.compact(items)}


def test_totals_count_what_is_missing_and_the_commonest_reasons() -> None:
    full = std.check(_full())
    no_chart = std.check(_full(stages=(_stage(1, "binance", 1, window=None),
                                       _stage(2, "hyperliquid", 2, window=None))))
    rows = [_row("a", full), _row("b", no_chart), _row("c", no_chart)]

    totals = std.totals(rows)
    chart = totals[4]

    assert (chart["key"], chart["missing"], chart["have"]) == ("chart", 2, 1)
    assert chart["common_reasons"][0]["count"] == 2
    assert chart["common_reasons"][0]["reason"].startswith("ряд денег ещё не построен")
    assert totals[0]["missing"] == 0
    assert std.most_common_missing(rows) == [("График", 2)]


def test_the_text_table_has_a_row_per_idea_and_a_totals_row() -> None:
    rows = [_row("alpha", std.check(_full())), _row("beta", std.check(_full(has_run=False,
                                                                           status="candidate")))]
    text = std.render_text({"rows": rows, "totals": std.totals(rows)})

    assert "alpha" in text and "beta" in text and "нет у скольких идей" in text
    assert "14/14" in text
    assert text.splitlines()[0].startswith("идея")


def test_the_morning_line_names_the_commonest_gap() -> None:
    no_chart = std.check(_full(stages=(_stage(1, "binance", 1, window=None),)))
    complete = std.check(_full())
    line = std.report_line({"rows": [_row("a", no_chart), _row("b", complete)]})

    assert line.startswith("стандарт страницы: у 1 из 2 идей не хватает пунктов")
    assert "«График»" in line
    assert std.report_line({"rows": [_row("b", complete)]}) is None


def test_ideas_put_away_and_calibration_books_are_skipped() -> None:
    class Idea:
        def __init__(self, idea_id: str, status: str) -> None:
            self.id = idea_id
            self.status = type("S", (), {"value": status})()

    assert std.skipped(Idea("old", "retired"))
    assert std.skipped(Idea("noise-book-1", "bench"))
    assert not std.skipped(Idea("alive", "bench"))
