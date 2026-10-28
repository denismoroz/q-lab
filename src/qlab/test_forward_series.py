"""The forward test's daily series (docs/FORWARD_CHART.md): what is stored, how a stage's chart
is assembled for the console and the morning report. A run is replayed in
`qlab.pipeline.test_forward_store`; the API route is tested in `qlab.api.test_forward_api`."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import forward_series as fs
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
)


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "series"


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _bars(values: list[float], start: str = "2026-03-01") -> pd.Series:
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="12h")
    return pd.Series(values, index=index)


# --------------------------------------------------------------------------
# The series itself


def test_bars_of_one_day_are_compounded_and_the_curve_starts_at_zero() -> None:
    returns = _bars([0.01, 0.01, -0.02, 0.0])  # two days, two bars each
    payload = fs.build_payload(trial_id=7, idea_id="a", venue="binance", interval="12h",
                               net_return=returns, hold_return=None)

    assert payload["points"][0] == ["2026-02-28", 0.0, None]  # the zero line, the eve of day 1
    assert [p[0] for p in payload["points"]] == ["2026-02-28", "2026-03-01", "2026-03-02"]
    assert payload["points"][1][1] == pytest.approx(1.01 * 1.01 - 1)
    assert payload["points"][2][1] == pytest.approx(1.01 * 1.01 * 0.98 - 1)
    assert payload["total_return"] == pytest.approx(payload["points"][-1][1])
    assert (payload["days"], payload["start"], payload["end"]) == (2, "2026-03-01", "2026-03-02")
    assert payload["max_dd"] == pytest.approx(0.98 - 1)


def test_the_yearly_figure_is_the_total_stretched_over_a_year() -> None:
    returns = _bars([0.001] * 20)  # ten days
    payload = fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="12h",
                               net_return=returns, hold_return=None)

    assert payload["days"] == 10
    stretched = (1 + payload["total_return"]) ** (365.25 / 10) - 1
    assert payload["ann_return"] == pytest.approx(stretched)


def test_the_hold_line_is_drawn_on_the_same_days_and_may_be_absent() -> None:
    returns = _bars([0.01, 0.0, 0.01, 0.0])
    hold = _bars([-0.02, 0.0, 0.0, 0.0])

    payload = fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="12h",
                               net_return=returns, hold_return=hold,
                               hold_coins=["BTC-SPOT"])

    assert payload["points"][1][2] == pytest.approx(-0.02)
    assert payload["hold_total"] == pytest.approx(-0.02)
    assert payload["hold_coins"] == ["BTC-SPOT"]


def test_a_series_says_whether_it_reproduces_the_trials_recorded_total() -> None:
    returns = _bars([0.01, 0.0, 0.01, 0.0])
    total = 1.01 * 1.01 - 1

    same = fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="12h",
                            net_return=returns, hold_return=None, recorded_total=total)
    other = fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="12h",
                             net_return=returns, hold_return=None, recorded_total=total + 0.01)
    unknown = fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="12h",
                               net_return=returns, hold_return=None)

    assert (same["matches_trial"], other["matches_trial"], unknown["matches_trial"]) \
        == (True, False, None)


def test_nothing_to_draw_is_an_error_not_an_empty_chart() -> None:
    with pytest.raises(ValueError, match="no returns"):
        fs.build_payload(trial_id=1, idea_id="a", venue=None, interval="1h",
                         net_return=pd.Series([], dtype=float), hold_return=None)


def test_a_stored_series_reads_back_and_a_missing_or_foreign_file_reads_as_none(store) -> None:
    payload = fs.build_payload(trial_id=42, idea_id="a", venue="binance", interval="12h",
                               net_return=_bars([0.01, 0.0]), hold_return=None)
    fs.save(payload, store)

    assert fs.load(42, store) == payload
    assert fs.load(43, store) is None
    (store / "44.json").write_text("not json", encoding="utf-8")
    assert fs.load(44, store) is None
    (store / "45.json").write_text('{"format": 99}', encoding="utf-8")
    assert fs.load(45, store) is None


def test_the_store_folder_follows_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QLAB_FORWARD_SERIES", str(tmp_path / "elsewhere"))

    assert fs.series_path(5) == tmp_path / "elsewhere" / "5.json"


# --------------------------------------------------------------------------
# What the console draws


def _idea(session, idea_id: str, status: IdeaStatus, parent: str | None = None) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=(f"Second stage of {parent} (docs/TWO_STAGE.md)" if parent else None)))
    session.flush()


def _run(session, idea_id: str, route: TrialRoute, source: str, *, started=None) -> int:
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref="x:Y")
    trial = repo.add_trial(session, spec_id=spec.id, config_hash=f"h-{idea_id}", code_sha="x",
                           params={}, status=TrialStatus.OK, route=route, route_reason="r",
                           source=TrialSource.IMPORTED,
                           started_at=started or datetime(2026, 10, 28, 2, tzinfo=UTC))
    return trial.id


def _stored(trial_id: int, store, idea_id: str = "x") -> None:
    fs.save(fs.build_payload(trial_id=trial_id, idea_id=idea_id, venue="binance", interval="12h",
                             net_return=_bars([0.01, 0.0]), hold_return=None), store)


def test_a_passed_idea_with_two_stages_shows_a_chart_per_stage(session, store) -> None:
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    _idea(session, "alpha-hl", IdeaStatus.VALIDATED, parent="alpha")
    t1 = _run(session, "alpha", TrialRoute.PAPER, "binance")
    t2 = _run(session, "alpha-hl", TrialRoute.PAPER, "hyperliquid")
    _stored(t1, store)
    _stored(t2, store)

    for page in ("alpha", "alpha-hl"):  # the parent's page and the child's show the same pair
        view = fs.forward_view(session, page, store=store)
        assert view["passed"] is True and view["two_stage"] is True
        assert [(s["stage"], s["idea_id"], s["venue"], s["trial_id"]) for s in view["stages"]] \
            == [(1, "alpha", "Binance", t1), (2, "alpha-hl", "Hyperliquid", t2)]
        assert all(s["series"] is not None and s["message"] is None for s in view["stages"])


def test_a_stage_whose_series_is_not_stored_says_so_in_plain_words(session, store) -> None:
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    _idea(session, "alpha-hl", IdeaStatus.VALIDATED, parent="alpha")
    t1 = _run(session, "alpha", TrialRoute.PAPER, "binance")
    _run(session, "alpha-hl", TrialRoute.PAPER, "hyperliquid")
    _stored(t1, store)

    first, second = fs.forward_view(session, "alpha", store=store)["stages"]

    assert first["series"] is not None
    assert second["series"] is None and "не сохранён" in second["message"]
    assert "forward-series rebuild" in second["message"]


def test_an_idea_without_a_second_stage_has_one_stage_and_says_so(session, store) -> None:
    _idea(session, "solo", IdeaStatus.VALIDATED)
    trial = _run(session, "solo", TrialRoute.PAPER, "hyperliquid")
    _stored(trial, store)

    view = fs.forward_view(session, "solo", store=store)

    assert view["passed"] is True and view["two_stage"] is False
    assert [s["stage"] for s in view["stages"]] == [1]


def test_a_second_stage_that_has_not_run_yet_is_a_message_not_a_chart(session, store) -> None:
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    _idea(session, "alpha-hl", IdeaStatus.CANDIDATE, parent="alpha")
    _stored(_run(session, "alpha", TrialRoute.PAPER, "binance"), store)

    second = fs.forward_view(session, "alpha", store=store)["stages"][1]

    assert second["trial_id"] is None and second["series"] is None
    assert "ещё нет прогона" in second["message"]


@pytest.mark.parametrize("status,route", [
    (IdeaStatus.BENCH, TrialRoute.NEEDS_FORWARD),
    (IdeaStatus.CANDIDATE, TrialRoute.REJECT),
    (IdeaStatus.REJECTED, TrialRoute.REJECT),
])
def test_an_idea_that_did_not_pass_gets_no_chart_and_a_reason(
        session, store, status, route) -> None:
    _idea(session, "beta", status)
    _stored(_run(session, "beta", route, "binance"), store)

    view = fs.forward_view(session, "beta", store=store)

    assert view["passed"] is False and view["stages"] == []
    assert view["reason"]


def test_a_retired_idea_gets_no_chart_even_if_its_last_run_passed(session, store) -> None:
    _idea(session, "gone", IdeaStatus.RETIRED)
    _stored(_run(session, "gone", TrialRoute.PAPER, "binance"), store)

    view = fs.forward_view(session, "gone", store=store)

    assert view["passed"] is False and "закрыта" in view["reason"]


def test_an_unknown_idea_is_none(session, store) -> None:
    assert fs.forward_view(session, "nobody", store=store) is None


def test_the_morning_report_lists_ideas_that_passed_on_the_day_once_per_strategy(
        session, store, monkeypatch) -> None:
    monkeypatch.setenv("QLAB_FORWARD_SERIES", str(store))
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    _idea(session, "alpha-hl", IdeaStatus.VALIDATED, parent="alpha")
    _idea(session, "old", IdeaStatus.VALIDATED)
    _idea(session, "retired", IdeaStatus.RETIRED)
    _stored(_run(session, "alpha", TrialRoute.PAPER, "binance"), store)
    _run(session, "alpha-hl", TrialRoute.PAPER, "hyperliquid")
    _run(session, "old", TrialRoute.PAPER, "binance", started=datetime(2026, 10, 20, 2, tzinfo=UTC))
    _run(session, "retired", TrialRoute.PAPER, "binance")

    found = fs.passed_on(session, datetime(2026, 10, 28, tzinfo=UTC),
                         datetime(2026, 10, 29, tzinfo=UTC))

    assert [e["idea_id"] for e in found] == ["alpha"]
    assert [(s["stage"], s["venue"], s["has_series"]) for s in found[0]["stages"]] \
        == [(1, "Binance", True), (2, "Hyperliquid", False)]


def test_passed_ideas_are_the_validated_ones_and_those_whose_last_run_passed(session) -> None:
    _idea(session, "v", IdeaStatus.VALIDATED)
    _idea(session, "p", IdeaStatus.PAPER)
    _idea(session, "w", IdeaStatus.BENCH)
    _idea(session, "r", IdeaStatus.RETIRED)
    _run(session, "p", TrialRoute.PAPER, "binance")
    _run(session, "w", TrialRoute.NEEDS_FORWARD, "binance")
    _run(session, "r", TrialRoute.PAPER, "binance")

    assert fs.passed_ideas(session) == ["p", "v"]


def test_rebuild_keeps_existing_series_and_reports_a_run_it_cannot_replay(session, store) -> None:
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    t1 = _run(session, "alpha", TrialRoute.PAPER, "binance")
    _stored(t1, store)
    _idea(session, "beta", IdeaStatus.VALIDATED)
    t2 = _run(session, "beta", TrialRoute.PAPER, "binance")

    done = fs.rebuild(session, store=store)

    assert (done[0][0], done[0][2]) == ("alpha", "уже есть")
    # beta's trial has no numbers or snapshot (an imported run): nothing honest to replay
    assert done[1][:2] == ("beta", t2) and done[1][2].startswith("нет:")
    assert fs.load(t2, store) is None


# --------------------------------------------------------------------------
# The morning report


def _use_session(monkeypatch, session) -> None:
    from contextlib import contextmanager

    @contextmanager
    def scope():
        yield session

    monkeypatch.setattr("qlab.registry.db.session_scope", scope)


def test_the_report_links_each_passed_idea_to_its_charts_and_says_what_is_ready(
        session, store, monkeypatch) -> None:
    from datetime import date

    from qlab import night

    monkeypatch.setenv("QLAB_FORWARD_SERIES", str(store))
    _use_session(monkeypatch, session)
    _idea(session, "alpha", IdeaStatus.VALIDATED)
    _idea(session, "alpha-hl", IdeaStatus.VALIDATED, parent="alpha")
    _stored(_run(session, "alpha", TrialRoute.PAPER, "binance"), store)
    _run(session, "alpha-hl", TrialRoute.PAPER, "hyperliquid")

    lines = night._forward_chart_lines(date(2026, 10, 28))

    assert lines[0] == f"## {night.FORWARD_CHARTS_TITLE}"
    entry = next(line for line in lines if "alpha" in line)
    assert f"{night.CONSOLE_URL}/ideas/alpha" in entry
    assert "этап 1 (Binance), этап 2 (Hyperliquid)" in entry
    assert "forward-series rebuild" in entry  # the second stage's series is not built yet


def test_the_report_has_no_section_when_nothing_passed_and_one_line_when_it_cannot_look(
        session, monkeypatch) -> None:
    from datetime import date

    from qlab import night

    _use_session(monkeypatch, session)
    assert night._forward_chart_lines(date(2026, 10, 28)) == []

    def broken():
        raise RuntimeError("no registry")

    monkeypatch.setattr("qlab.registry.db.session_scope", broken)
    lines = night._forward_chart_lines(date(2026, 10, 28))
    assert "не посчитано" in " ".join(lines).lower()
