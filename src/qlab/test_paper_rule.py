"""Tests for `qlab.paper_rule`: the owner's "at least 4-6 weeks on paper" rule."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab import paper_rule
from qlab.registry import repo
from qlab.registry.models import AssetClass, Base, Profile, SourceType

HOUR_MS = 3_600_000

# What `qlab.night.paper_reconciliation` saves, from the night of 2026-10-25.
REPLAY_LINES = [
    "**Bv2 (book 3 = b2, sized as created with carry on; book 4 = b2_cold)**",
    "    book 3 (headroom 1.0, sized with carry: True)",
    "      matched hours 3980, price mismatches 0",
    "      paper  +4.11%   replay +4.11%",
    "      max |hourly equity diff| 0.0000",
    "      daily return correlation 1.0000 (n=42)",
    "    book 4 (headroom 1.5, sized with carry: False)",
    "      matched hours 3976, price mismatches 0",
    "      paper  +5.53%   replay +9.80%",
    "      max |hourly equity diff| 0.0000",
    "      daily return correlation 0.7000 (n=42)",
    "**trend**",
    "    matched hours 969 (2026-09-14 14:00:00+00:00 .. 2026-10-25 01:00:00+00:00)",
    "    paper  2999.29 -> 2963.24 (-1.20%)",
    "    replay 2999.29 -> 2962.79 (-1.22%)",
    "    daily return correlation 0.9998 (n=41)",
    "    fills replay 285",
    "**Бумага и стенд за один отрезок**",
    "    Тренд: за 14.09–24.10 бумага −1.2%, стенд +3.6%",
]


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _export(folder: Path, first: date, days: int) -> None:
    """A night's export: Bv2 books 3 and 4 (two coins each) and the trend book,
    hourly from `first` for `days` days."""
    folder.mkdir(parents=True, exist_ok=True)
    ts = [_ms(first) + h * HOUR_MS for h in range(24 * days + 1)]
    rows = [{"strategy_id": sid, "ts_ms": t, "coin": coin, "equity": 100.0}
            for sid in (3, 4) for t in ts for coin in ("BTC", "ETH")]
    pd.DataFrame(rows).to_csv(folder / "b2_equity.csv", index=False)
    pd.DataFrame({"strategy_id": 5, "ts_ms": ts, "equity": 3000.0}).to_csv(
        folder / "trend_equity.csv", index=False)


def _books() -> dict:
    return {
        "bv2-a": {"book": "b2", "label": "Bv2, запас 1.0", "paper_file": "b2_equity.csv",
                  "strategy_id": 3, "sum_coins": True, "replay_block": "book 3"},
        "bv2-b": {"book": "b2_cold", "label": "Bv2, запас 1.5", "paper_file": "b2_equity.csv",
                  "strategy_id": 4, "sum_coins": True, "replay_block": "book 4"},
        "trend-a": {"book": "trend", "label": "Тренд", "paper_file": "trend_equity.csv",
                    "strategy_id": 5, "sum_coins": False, "replay_block": "trend"},
    }


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    for idea_id in _books():
        repo.upsert_idea(s, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    yield s
    s.close()
    engine.dispose()


def test_the_replay_scripts_output_is_read_per_book() -> None:
    blocks = paper_rule.parse_replay_lines(REPLAY_LINES)

    assert set(blocks) == {"book 3", "book 4", "trend"}
    assert blocks["book 3"].paper == pytest.approx(0.0411)
    assert blocks["book 3"].replay == pytest.approx(0.0411)
    assert blocks["book 3"].correlation == 1.0
    assert blocks["trend"].paper == pytest.approx(-0.0120)
    assert blocks["trend"].replay == pytest.approx(-0.0122)
    assert blocks["book 3"].matches() is True
    assert blocks["book 4"].matches() is False  # correlation 0.70, under the preregistered 0.80
    assert blocks["trend"].matches() is True  # 0.02 points apart, correlation 0.9998
    assert paper_rule.ReplayFigures().matches() is None
    assert paper_rule.parse_replay_lines(["export from the production host failed: x"]) == {}


def test_the_retuned_books_blocks_are_read_each_under_its_own_name() -> None:
    lines = [
        *REPLAY_LINES[:-2],  # the two older books' blocks, and the trend block
        "**Bv2 re-tuned monthly (book 8 = b2_retune)**",
        "    book 8 (b2_retune: headroom 1.0 or 1.5 chosen monthly; live from 2026-10-25 14:00)",
        "      matched hours 1, price mismatches 0",
        "      paper and replay totals: not yet, no live hour has closed after the launch point",
        "      daily return correlation not computed: n=0 complete UTC days, at least 14 needed",
        "      history before launch: 3518 hours both sides, paper +7.96% replay +7.96%, "
        "max |hourly equity diff| 0.0000 (0.00% of equity)",
        "**trend_retune_3k**",
        "    replay of strategy 6: history 662 daily bars to 2026-10-25, replay through "
        "2026-10-25 13:00 UTC (paper's last bar 2026-10-25 13:00)",
        "      matched hours 2",
        "      paper  2999.64 -> 2999.40 (-0.01%)",
        "      replay 2999.64 -> 2999.40 (-0.01%)",
        "      max |hourly equity diff| 0.0000",
        "      daily return correlation not computed: n=0 complete UTC days, at least 14 needed",
    ]

    blocks = paper_rule.parse_replay_lines(lines)

    assert set(blocks) == {"book 3", "book 4", "trend", "book 8", "trend_retune_3k"}
    # the old trend block keeps its own figures: the new header does not start it again
    assert blocks["trend"].correlation == pytest.approx(0.9998)
    assert blocks["trend_retune_3k"].paper == pytest.approx(-0.0001)
    assert blocks["trend_retune_3k"].replay == pytest.approx(-0.0001)
    # no correlation yet: unknown, never a match
    assert blocks["trend_retune_3k"].correlation is None
    assert blocks["trend_retune_3k"].matches() is None
    assert blocks["book 8"].paper is None and blocks["book 8"].matches() is None


def test_a_correlation_on_a_few_days_is_unknown_not_a_match() -> None:
    lines = ["**trend_retune_3k**", "    paper  3000.00 -> 3003.00 (+0.10%)",
             "    replay 3000.00 -> 3003.00 (+0.10%)",
             "    daily return correlation 1.0000 (n=3)"]

    figures = paper_rule.parse_replay_lines(lines)["trend_retune_3k"]

    assert figures.correlation == 1.0 and figures.days == 3
    assert figures.matches() is None
    lines[-1] = "    daily return correlation 0.9000 (n=20)"
    assert paper_rule.parse_replay_lines(lines)["trend_retune_3k"].matches() is True


def test_a_book_with_a_young_replay_is_not_ready_even_after_four_weeks(session, tmp_path) -> None:
    root = tmp_path / "paper"
    _export(root / "2026-10-25", date(2026, 9, 13), 29)
    lines = ["**trend**", "    paper  1.00 -> 1.01 (+1.00%)", "    replay 1.00 -> 1.01 (+1.00%)",
             "    daily return correlation 1.0000 (n=2)"]

    rule = paper_rule.paper_rule(
        session, "trend-live", {"book": "trend", "paper_file": "trend_equity.csv",
                                 "strategy_id": 5, "replay_block": "trend"},
        root=root, replay_lines=lines)

    assert rule.weeks >= paper_rule.MIN_WEEKS
    assert rule.replay == "unknown" and rule.ready is False


def test_a_low_correlation_is_not_a_match() -> None:
    # The preregistered bar is ">= 0.80" (seed/preregistration/2026-09-21-paper-bv2-trend.yaml).
    assert paper_rule.ReplayFigures(paper=0.01, replay=0.01, correlation=0.79).matches() is False
    assert paper_rule.ReplayFigures(paper=0.01, replay=0.01, correlation=0.80).matches() is True
    # Totals alone decide nothing: no bar on them was ever declared.
    assert paper_rule.ReplayFigures(paper=0.01, replay=0.01).matches() is None


def test_age_runs_from_the_first_row_to_the_latest(tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 42)

    first, last = paper_rule.book_age(tmp_path / "2026-10-25" / "b2_equity.csv",
                                      strategy_id=4, sum_coins=True)

    assert first == datetime(2026, 9, 13, tzinfo=UTC)
    assert last - first == timedelta(days=42)


def test_age_counts_only_rows_from_realised_from(tmp_path) -> None:
    # frab's b2_retune replays its history at launch; those rows are not weeks on paper.
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 42)

    first, last = paper_rule.book_age(tmp_path / "2026-10-25" / "b2_equity.csv",
                                      strategy_id=4, sum_coins=True,
                                      realised_from=datetime(2026, 10, 18, tzinfo=UTC))

    assert first == datetime(2026, 10, 18, tzinfo=UTC)
    assert last - first == timedelta(days=7)


def test_a_book_is_ready_after_four_weeks_with_the_replay_matching(session, tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 42)

    rules, problems = paper_rule.paper_rules(session, books=_books(), root=tmp_path,
                                             replay_lines=REPLAY_LINES)

    by_idea = {r.idea_id: r for r in rules}
    assert problems == []
    assert by_idea["bv2-a"].weeks == pytest.approx(6.0)
    assert (by_idea["bv2-a"].replay, by_idea["bv2-a"].ready) == ("match", True)
    # Old enough, but the replay's daily correlation is under the preregistered bar: not ready.
    assert (by_idea["bv2-b"].replay, by_idea["bv2-b"].ready) == ("mismatch", False)
    assert (by_idea["trend-a"].replay, by_idea["trend-a"].ready) == ("match", True)
    assert paper_rule.READY_TEXT in by_idea["bv2-a"].line()
    assert by_idea["bv2-a"].line().startswith(
        "Bv2, запас 1.0: на бумаге 6 недель из минимум 4–6 (с 13.09.2026); "
        "повтор бумаги на стенде сошёлся; по правилу владельца книга готова к решению")
    assert "срок вышел, но к решению книга не готова" in by_idea["bv2-b"].line()


def test_a_young_book_is_not_ready_even_when_the_replay_matches(session, tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 10, 3), 22)  # three weeks and a day

    rules, _ = paper_rule.paper_rules(session, books=_books(), root=tmp_path,
                                      replay_lines=REPLAY_LINES)

    rule = next(r for r in rules if r.idea_id == "bv2-a")
    assert (rule.whole_weeks, rule.replay, rule.ready) == (3, "match", False)
    assert rule.line().endswith("до решения ещё рано")
    assert "из минимум 4–6" in rule.line()


def test_exactly_four_weeks_is_enough(session, tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 9, 27), 28)

    rules, _ = paper_rule.paper_rules(session, books=_books(), root=tmp_path,
                                      replay_lines=REPLAY_LINES)

    assert next(r for r in rules if r.idea_id == "trend-a").ready


def test_without_any_replay_figures_the_state_is_unknown_and_not_ready(session, tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 42)

    rules, _ = paper_rule.paper_rules(session, books=_books(), root=tmp_path, replay_lines=[])

    assert {r.replay for r in rules} == {"unknown"} and not any(r.ready for r in rules)
    assert "повтор бумаги на стенде не посчитан" in rules[0].line()


def test_a_missing_export_is_a_line_not_a_failure(session, tmp_path) -> None:
    (tmp_path / "2026-10-25").mkdir()
    books = {"trend-a": _books()["trend-a"]}

    lines = paper_rule.report_lines(session, books=books, root=tmp_path, replay_lines=[])

    assert lines[0] == f"**{paper_rule.RULE_TITLE}**"
    assert lines[1] == "    Тренд: нет выгрузки бумажной книги, срок не посчитан"


def test_the_latest_export_holding_the_file_is_used(tmp_path) -> None:
    _export(tmp_path / "2026-10-24", date(2026, 9, 13), 40)
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 41)
    (tmp_path / "2026-10-25-afternoon").mkdir()  # holds other files only
    (tmp_path / "2026-10-25-afternoon" / "qx_b2_equity.csv").write_text("x\n")

    assert paper_rule.latest_export(tmp_path, "b2_equity.csv") == \
        tmp_path / "2026-10-25" / "b2_equity.csv"
    assert paper_rule.latest_export(tmp_path, "nothing.csv") is None


def test_the_newest_night_with_replay_lines_is_found(tmp_path) -> None:
    (tmp_path / "2026-10-23.json").write_text(json.dumps({"paper": ["old"]}))
    (tmp_path / "2026-10-24.json").write_text(json.dumps({"paper": ["new"]}))
    (tmp_path / "2026-10-25.json").write_text(json.dumps({"results": []}))  # no paper step yet

    assert paper_rule.latest_replay_lines(tmp_path) == ["new"]
    assert paper_rule.latest_replay_lines(tmp_path / "nowhere") == []


def test_the_idea_page_rule_is_none_for_an_idea_without_a_book(session, tmp_path) -> None:
    _export(tmp_path / "2026-10-25", date(2026, 9, 13), 42)

    assert paper_rule.paper_rule_for_idea(session, "nobody", books=_books(), root=tmp_path) is None
    rule = paper_rule.paper_rule_for_idea(session, "bv2-a", books=_books(), root=tmp_path,
                                          night_dir=tmp_path / "no-nights")
    assert rule is not None and rule.replay == "unknown"
    body = rule.to_dict()
    assert body["whole_weeks"] == 6 and body["min_weeks"] == 4 and body["ready"] is False


def test_russian_plurals_of_weeks() -> None:
    words = [paper_rule._weeks_word(n) for n in (0, 1, 2, 4, 5, 11, 12, 21, 22)]
    assert words == ["недель", "неделя", "недели", "недели", "недель", "недель", "недель",
                     "неделя", "недели"]
