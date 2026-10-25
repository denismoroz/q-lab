"""Tests for `qlab.decay`: the realised results against the stand's own history."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlab import decay, night, paper_check, paper_rule
from qlab.decay import Band, check_band

HOUR_MS = 3_600_000


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _stand_returns(values: list[float], start: date = date(2026, 1, 1)) -> pd.Series:
    """A daily net_return series indexed by decision time, like the harness's."""
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="D")
    return pd.Series(values, index=index)


def _paper_curve(start: datetime, hours: int, hourly_return: float) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start), periods=hours + 1, freq="h")
    return pd.Series(1000.0 * (1 + hourly_return) ** np.arange(hours + 1), index=index)


# --- the distribution of same-length windows ------------------------------------------------


def test_same_length_windows_of_a_steady_series_all_return_the_same() -> None:
    closes = pd.Series(100.0 * 1.01 ** np.arange(31),
                       index=pd.date_range("2026-01-01", periods=31, freq="D", tz="UTC"))

    windows = decay.rolling_windows(closes, 10)

    assert len(windows) == 21  # 31 points, windows of 10 steps: starts 0..20
    assert all(w.total == pytest.approx(1.01 ** 10 - 1) for w in windows)
    assert all(w.max_dd == 0.0 for w in windows)
    assert all(w.spread == pytest.approx(0.0, abs=1e-12) for w in windows)
    assert windows[0].start == date(2026, 1, 1)


def test_windows_measure_return_fall_and_spread_of_a_known_path() -> None:
    closes = pd.Series([100.0, 110.0, 99.0, 108.9, 108.9],
                       index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"))

    first, second = decay.rolling_windows(closes, 3)

    assert first.total == pytest.approx(108.9 / 100 - 1)
    assert first.max_dd == pytest.approx(99 / 110 - 1)  # the fall from the 110 peak
    returns = np.array([0.10, -0.10, 0.10])
    assert first.spread == pytest.approx(returns.std(ddof=1))
    assert second.total == pytest.approx(108.9 / 110 - 1)
    # A window longer than the path has no windows.
    assert decay.rolling_windows(closes, 4) == decay.rolling_windows(closes, 4)[:1]
    assert decay.rolling_windows(closes, 5) == []


def test_daily_closes_take_the_last_value_of_the_day_and_midnight_closes_the_day_before() -> None:
    index = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 23:00", "2026-01-02 00:00",
                            "2026-01-02 12:00", "2026-01-04 00:00"], utc=True)
    equity = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

    closes = decay.daily_closes(equity)

    # 01-01 closes at 3.0 (the midnight row belongs to it), 01-02 at 4.0, and the row at
    # 01-04 00:00 closes 01-03 at 5.0.
    assert list(closes.index.strftime("%m-%d")) == ["01-01", "01-02", "01-03"]
    assert list(closes) == [3.0, 4.0, 5.0]
    # A day without any row holds the previous close.
    gap = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01 12:00", "2026-01-03 12:00"],
                                                      utc=True))
    assert list(decay.daily_closes(gap)) == [1.0, 1.0, 2.0]


def test_percentile_counts_ties_as_half_and_an_empty_sample_has_none() -> None:
    assert decay.percentile_of(5.0, [1, 2, 3, 4]) == 100.0
    assert decay.percentile_of(0.0, [1, 2, 3, 4]) == 0.0
    assert decay.percentile_of(2.5, [1, 2, 3, 4]) == 50.0
    assert decay.percentile_of(2.0, [1, 2, 2, 3]) == 50.0  # one below, two tied: (1 + 1) / 4
    assert decay.percentile_of(1.0, []) is None


def test_the_spread_is_the_stands_tenth_median_and_ninetieth() -> None:
    spread = decay.spread_of(list(np.arange(101) / 100))
    assert (spread.p10, spread.p50, spread.p90) == pytest.approx((0.1, 0.5, 0.9))
    assert decay.spread_of([]) is None


def test_a_strategy_still_warming_up_has_no_history_to_describe() -> None:
    returns = _stand_returns([0.0, 0.0, 0.0, 0.01, 0.0, -0.01])
    assert len(decay.trim_flat_start(returns)) == 3
    assert decay.trim_flat_start(_stand_returns([0.0, 0.0])).empty


# --- the comparison of one book ---------------------------------------------------------------


def _compare(realised, stand, *, bands=None, alert=None, **kw):
    return decay.compare(idea_id="x", label="X", book="b", live=False, realised=realised,
                         stand_net_return=stand, bands=bands or [], alert_percentile=alert, **kw)


def test_a_flat_realised_book_sits_at_the_bottom_of_a_rising_stand() -> None:
    # The stand rose 1% a day for 40 days; the book was flat for a week.
    stand = _stand_returns([0.01] * 40, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 7, 0.0)

    result = _compare(realised, stand)

    assert result.window_days == 7
    assert result.ret.realised == pytest.approx(0.0)
    assert result.ret.stand.p50 == pytest.approx(1.01 ** 7 - 1)
    assert result.ret.percentile == 0.0  # every stand window beat it
    # 40 days of history, 7-day windows: 34 of them, 5 not overlapping
    assert result.windows == 34 and result.history_days == 40 and result.independent == 5
    assert result.history_to == date(2026, 2, 9)  # the close of the 40th day
    assert result.drawdown.realised == 0.0


def test_the_stand_history_ends_where_the_realised_period_starts() -> None:
    # The stand's last 20 days collapse; the book starts before them, so they are not "history".
    values = [0.01] * 40 + [-0.05] * 20
    stand = _stand_returns(values, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 2, 10, tzinfo=UTC), 24 * 5, 0.0)

    result = _compare(realised, stand)

    assert result.history_to <= date(2026, 2, 10)
    assert result.ret.stand.p10 > 0  # no collapse window leaked into the distribution


def test_realised_drawdown_and_daily_spread_are_measured_on_daily_closes() -> None:
    stand = _stand_returns(list(np.tile([0.02, -0.01], 30)), date(2026, 1, 1))
    index = pd.date_range("2026-03-01", periods=6 * 24 + 1, freq="h", tz="UTC")
    # Equity falls 10% on day 3, otherwise flat.
    equity = pd.Series(1000.0, index=index)
    equity[equity.index >= pd.Timestamp("2026-03-03 12:00", tz="UTC")] = 900.0

    result = _compare(equity, stand)

    assert result.drawdown.realised == pytest.approx(-0.10)
    assert result.spread.realised > 0
    assert result.ret.realised == pytest.approx(-0.10)


def test_too_little_stand_history_gives_no_percentile_and_says_so() -> None:
    stand = _stand_returns([0.01] * 7, date(2026, 1, 1))  # one 7-day window: no spread to compare
    realised = _paper_curve(datetime(2026, 2, 1, tzinfo=UTC), 24 * 7, 0.0001)

    result = _compare(realised, stand)

    assert result.ret.percentile is None and result.ret.stand is None
    assert "не хватает" in result.note
    assert result.alert is None
    assert any("сравнивать не с чем" in line for line in decay.result_lines(result))


def test_a_book_with_one_row_has_no_result() -> None:
    stand = _stand_returns([0.01] * 40)
    one = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 0, 0.0)
    assert _compare(one, stand) is None


def test_realised_from_cuts_a_longer_book_down_to_its_period() -> None:
    stand = _stand_returns([0.01] * 60, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 20, 0.0005)

    result = _compare(realised, stand, realised_from=datetime(2026, 3, 11, tzinfo=UTC))

    assert result.first == datetime(2026, 3, 11, tzinfo=UTC)
    assert result.window_days == 10


# --- preregistered bands ----------------------------------------------------------------------


BAND = Band(id="year", metric="return", label="year band", source="seed/x.yaml", lo=-0.132,
            hi=0.227, horizon_days=365)
STOP = Band(id="stop", metric="max_drawdown", label="stop", source="seed/x.yaml", kind="stop",
            lo=-0.20)


def test_a_band_is_inside_outside_or_not_applicable_to_a_shorter_period() -> None:
    assert check_band(BAND, value=0.10, realised_days=400).status == "inside"
    assert check_band(BAND, value=0.30, realised_days=400).status == "outside"
    assert check_band(BAND, value=-0.20, realised_days=400).status == "outside"
    # A one-year band says nothing about six weeks.
    assert check_band(BAND, value=0.30, realised_days=42).status == "not_applicable"
    assert check_band(STOP, value=-0.07, realised_days=42).status == "inside"
    assert check_band(STOP, value=-0.25, realised_days=42).status == "outside"
    # An open-ended band has only one end.
    half = Band(id="h", metric="return", label="h", source="s", hi=0.04)
    assert check_band(half, value=0.039, realised_days=1).status == "inside"
    assert check_band(half, value=0.041, realised_days=1).status == "outside"


def test_bands_reach_the_result_and_the_report_says_what_they_mean() -> None:
    stand = _stand_returns([0.001, -0.001] * 40, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 10, 0.0001)

    result = _compare(realised, stand, bands=[BAND, STOP])

    assert [b.status for b in result.bands] == ["not_applicable", "inside"]
    text = "\n".join(decay.result_lines(result))
    assert "к этому отрезку не относится (заявлено на 365 дней)" in text
    assert "условие остановки не сработало" in text

    deep = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 10, -0.01)
    fired = _compare(deep, stand, bands=[STOP])
    assert fired.bands[0].status == "outside"
    assert "условие остановки СРАБОТАЛО" in "\n".join(decay.result_lines(fired))


# --- no declared threshold, no label -----------------------------------------------------------


def test_without_a_declared_threshold_nothing_is_labelled() -> None:
    assert decay.alert_label(3.0, None) is None
    assert decay.alert_label(None, 10.0) is None
    assert decay.alert_label(12.0, 10.0) is None
    assert decay.alert_label(10.0, 10.0) == "below"
    assert decay.alert_label(2.0, 10.0) == "below"

    stand = _stand_returns([0.01] * 40, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 7, 0.0)  # bottom of the stand
    assert _compare(realised, stand).alert is None
    flagged = _compare(realised, stand, alert=10.0)
    assert flagged.alert == "below"
    assert "ниже порога тревоги" in "\n".join(decay.result_lines(flagged))
    assert "ниже порога тревоги" not in "\n".join(decay.result_lines(_compare(realised, stand)))


def test_the_threshold_is_null_or_a_number_from_zero_to_a_hundred(tmp_path) -> None:
    for bad in ("alert_percentile: 150", "alert_percentile: abc", "alert_percentile: -1"):
        path = tmp_path / "decay.yaml"
        path.write_text(bad + "\n")
        with pytest.raises(ValueError, match="alert_percentile"):
            decay.load_config(path)
    path.write_text("alert_percentile: 10\n")
    assert decay.load_config(path)["alert_percentile"] == 10


def test_the_report_text_is_plain_russian_with_no_percentile_jargon() -> None:
    stand = _stand_returns([0.001, -0.001, 0.002] * 30, date(2026, 1, 1))
    realised = _paper_curve(datetime(2026, 3, 1, tzinfo=UTC), 24 * 10, 0.0001)

    text = "\n".join(decay.result_lines(_compare(realised, stand)))

    assert "из 100" in text and "обычно на стенде" in text
    for jargon in ("перцентиль", "percentile", "квантиль", "дисперсия", "волатильность"):
        assert jargon not in text


# --- the live book's hourly export ------------------------------------------------------------


def test_the_live_export_is_one_bounded_hourly_sample() -> None:
    sql = decay.live_export_sql(1, 1_760_000_000_000)
    assert "from equity_snapshots" in sql and "strategy_id=1" in sql
    assert "ts_ms>=1760000000000" in sql and "ts_ms%3600000<60000" in sql
    assert "order by ts_ms" in sql


def test_live_books_are_kept_apart_from_paper_books(tmp_path) -> None:
    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n  p:\n    book: trend\n    paper_file: t.csv\n"
        "live_books:\n  frab:\n    book: two_phase\n    paper_file: frab_equity.csv\n"
        "    strategy_id: 1\n    realised_from: 2026-10-10\n", encoding="utf-8")

    assert set(paper_check.load_books(books)) == {"p"}  # the paper checks never see FRAB
    everything = decay.load_all_books(books)
    assert everything["p"]["live"] is False and everything["frab"]["live"] is True
    [(filename, sql)] = decay.live_exports(everything)
    assert filename == "frab_equity.csv"
    assert f"ts_ms>={_ms(date(2026, 10, 10))}" in sql


def test_the_shipped_mapping_and_config_are_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    books = decay.load_all_books(root / "night" / "paper_books.yaml")
    assert books["frab"]["live"] and books["frab"]["strategy_id"] == 1
    for cfg in books.values():
        assert (root / cfg["spec"]).is_file()

    config = decay.load_config(root / "night" / "decay.yaml")
    assert config["alert_percentile"] is None  # the owner sets it; nothing is invented
    for idea in config["preregistered"]:
        assert idea in books
        for band in decay.bands_for(config, idea):
            assert (root / band.source.split(",")[0].strip()).is_file()


# --- the night step on fixture exports --------------------------------------------------------


def _trend_export(folder: Path, start: datetime, hours: int, hourly: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    start_ms = int(start.timestamp() * 1000)
    pd.DataFrame({
        "strategy_id": 5, "ts_ms": [start_ms + h * HOUR_MS for h in range(hours + 1)],
        "equity": [3000.0 * (1 + hourly) ** h for h in range(hours + 1)],
    }).to_csv(folder / "trend_equity.csv", index=False)


def _books(tmp_path: Path) -> Path:
    path = tmp_path / "books.yaml"
    path.write_text(
        "books:\n  trend-x:\n    book: trend\n    spec: x.yaml\n    label: Тренд\n"
        "    paper_file: trend_equity.csv\n    strategy_id: 5\n    sum_coins: false\n"
        "live_books:\n  frab-x:\n    book: two_phase\n    spec: y.yaml\n    label: FRAB\n"
        "    paper_file: frab_equity.csv\n    equity_column: total_equity\n    strategy_id: 1\n"
        "    realised_from: 2026-10-10\n", encoding="utf-8")
    return path


def test_the_night_step_reads_the_exports_and_reports_each_book(tmp_path, monkeypatch) -> None:
    root = tmp_path / "paper"
    _trend_export(root / "2026-10-25", datetime(2026, 9, 14, tzinfo=UTC), 24 * 30, 0.0001)
    # The live book's export has its own column and a long history; only the period counts.
    live_ms = _ms(date(2026, 10, 1))
    pd.DataFrame({"strategy_id": 1,
                  "ts_ms": [live_ms + h * HOUR_MS for h in range(24 * 24)],
                  "total_equity": [500.0 + 0.001 * h for h in range(24 * 24)]}
                 ).to_csv(root / "2026-10-25" / "frab_equity.csv", index=False)
    seen = []

    def fake_stand(session, spec_path, *, end, cut):
        seen.append((spec_path, end, cut))
        return _stand_returns([0.001, -0.0005, 0.002] * 60, date(2026, 1, 1))

    config = {"alert_percentile": None, "preregistered": {}}
    results, lines = decay.run_decay(None, date(2026, 10, 25), date(2026, 10, 24),
                                     books=decay.load_all_books(_books(tmp_path)), config=config,
                                     root=root, stand_fn=fake_stand)

    assert [r.idea_id for r in results] == ["trend-x", "frab-x"]
    # The stand is asked for its history up to the first realised row of each book.
    assert seen[0][2] == pd.Timestamp("2026-09-14", tz="UTC")
    assert seen[1][2] == pd.Timestamp("2026-10-10", tz="UTC")
    frab = results[1]
    assert frab.live and frab.first == datetime(2026, 10, 10, tzinfo=UTC)
    text = "\n".join(lines)
    assert "Тренд (бумага, 30 дней с 14.09)" in text
    assert "FRAB (живые деньги, 15 дней с 10.10)" in text
    assert "`alert_percentile`" in lines[-1]
    assert "«хорошо» и «плохо» здесь не говорится" in lines[-1]


def test_a_missing_export_or_stand_history_is_a_line_not_a_failure(tmp_path) -> None:
    root = tmp_path / "paper"
    _trend_export(root / "2026-10-25", datetime(2026, 9, 14, tzinfo=UTC), 24 * 10, 0.0001)

    def broken(session, spec_path, *, end, cut):
        raise decay.StandHistoryMissing("the panel of x.yaml is not cached")

    results, lines = decay.run_decay(None, date(2026, 10, 25), date(2026, 10, 24),
                                     books=decay.load_all_books(_books(tmp_path)),
                                     config={"alert_percentile": None}, root=root,
                                     stand_fn=broken)

    assert results == []
    assert any(ln.startswith("Тренд: на стенде нет истории для сравнения") for ln in lines)
    assert any(ln.startswith("FRAB: нет выгрузки книги (frab_equity.csv)") for ln in lines)


def _use_registry(tmp_path, monkeypatch, *, tables: bool):
    import qlab.registry.db as db_module
    from qlab.registry.models import Base, DecayCheck

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    if tables:
        Base.metadata.create_all(db_module.get_engine())
    else:  # a registry from before the migration
        Base.metadata.create_all(
            db_module.get_engine(),
            tables=[t for n, t in Base.metadata.tables.items() if n != DecayCheck.__tablename__])
    return db_module


def _seed_night(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    books = _books(tmp_path)
    monkeypatch.setattr(paper_check, "BOOKS", books)
    monkeypatch.setattr(paper_rule, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(decay, "CONFIG", tmp_path / "decay.yaml")
    (tmp_path / "decay.yaml").write_text("alert_percentile: null\npreregistered: {}\n")
    monkeypatch.setattr(decay, "stand_net_return",
                        lambda session, spec_path, *, end, cut:
                        _stand_returns([0.001, -0.0005, 0.002] * 60, date(2026, 1, 1)))
    _trend_export(tmp_path / "paper" / "2026-10-25", datetime(2026, 9, 14, tzinfo=UTC), 24 * 30,
                  0.0001)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    return books


def _idea(session, idea_id: str) -> None:
    from qlab.registry import repo
    from qlab.registry.models import AssetClass, IdeaStatus, Profile, SourceType

    repo.upsert_idea(session, id=idea_id, title="x", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                     status=IdeaStatus.PAPER, notes="")


def test_the_night_writes_the_results_and_the_report_section(tmp_path, monkeypatch) -> None:
    db_module = _use_registry(tmp_path, monkeypatch, tables=True)
    _seed_night(tmp_path, monkeypatch)
    from qlab.registry.db import session_scope
    from qlab.registry.models import DecayCheck

    with session_scope() as session:
        _idea(session, "trend-x")
    day = date(2026, 10, 25)
    state = {"done": {}}

    night.decay_stage(day, date(2026, 10, 24), state)

    assert len(state["decay"]["results"]) == 1
    with session_scope() as session:
        row = session.query(DecayCheck).one()
        assert row.idea_id == "trend-x" and row.at == day
        assert row.return_percentile == state["decay"]["results"][0]["ret"]["percentile"]
        # The same night again replaces the row instead of adding one.
        night.decay_stage(day, date(2026, 10, 24), state)
    with session_scope() as session:
        assert session.query(DecayCheck).count() == 1
        latest = decay.latest_decay(session, "trend-x")
        assert latest["window_days"] == 30 and latest["at"] == "2026-10-25"
        assert decay.latest_decay(session, "no-such-idea") is None

    report = night.write_report(day, state).read_text(encoding="utf-8")
    assert "## Ведёт ли себя стратегия как на стенде" in report
    assert "Тренд (бумага, 30 дней с 14.09)" in report
    db_module._engine = None
    db_module._SessionLocal = None


def test_before_the_migration_the_result_stays_in_the_nights_state(tmp_path, monkeypatch) -> None:
    db_module = _use_registry(tmp_path, monkeypatch, tables=False)
    _seed_night(tmp_path, monkeypatch)
    from qlab.registry.db import session_scope

    day = date(2026, 10, 25)
    state = {"done": {}}
    night.decay_stage(day, date(2026, 10, 24), state)

    assert any("`decay_check` ещё нет" in ln for ln in state["decay"]["lines"])
    night.save_state(day, state)
    monkeypatch.setattr(decay, "NIGHT_DIR", night.STATE_DIR)
    with session_scope() as session:
        found = decay.latest_decay(session, "trend-x")
    assert found is not None and found["window_days"] == 30
    assert json.loads((night.STATE_DIR / "2026-10-25.json").read_text())["decay"]["results"]
    db_module._engine = None
    db_module._SessionLocal = None


def test_a_failing_decay_step_is_a_report_line_not_a_failed_night(tmp_path, monkeypatch) -> None:
    db_module = _use_registry(tmp_path, monkeypatch, tables=True)
    _seed_night(tmp_path, monkeypatch)
    monkeypatch.setattr(decay, "run_decay", lambda *a, **kw: 1 / 0)

    state = {"done": {}}
    night.decay_stage(date(2026, 10, 25), date(2026, 10, 24), state)

    assert state["decay"]["results"] == []
    assert state["decay"]["lines"][0].startswith("не посчитано: ZeroDivisionError")
    db_module._engine = None
    db_module._SessionLocal = None


def test_live_books_are_exported_by_the_paper_reconciliation(tmp_path, monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(paper_check, "BOOKS", _books(tmp_path))
    monkeypatch.setattr(night, "PAPER_DIR", tmp_path / "paper")
    asked = []
    monkeypatch.setattr(night, "_ssh_query", lambda sql, out, header=True: asked.append((sql, out)))
    monkeypatch.setattr(night.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="", stderr=""))

    night.paper_reconciliation(date(2026, 10, 25))

    live = [(sql, out) for sql, out in asked if "equity_snapshots" in sql]
    assert len(live) == 1 and live[0][1].name == "frab_equity.csv"
    assert "strategy_id=1" in live[0][0]


def test_the_stand_history_is_cut_before_the_strategy_runs(tmp_path, monkeypatch) -> None:
    """The strategy must never see a bar after the first realised row, and no
    trial is written: the panel is cached, the run is a description."""
    from qlab.harness.panel import MarketPanel  # noqa: F401 - the real class the cache returns

    index = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    prices = pd.DataFrame({"BTC": 100.0 * 1.01 ** np.arange(100)}, index=index)
    zero = pd.DataFrame(0.0, index=index, columns=["BTC"])
    seen_rows = []

    class Strategy:
        def target_weights(self, panel, params):
            seen_rows.append(len(panel.prices))
            return pd.DataFrame(1.0, index=panel.prices.index, columns=["BTC"])

    panel = MarketPanel(
        snapshot_id="s", prices=prices, funding=zero, tradeable=pd.DataFrame(True, index=index,
                                                                         columns=["BTC"]),
        volume=zero + 1.0, high=prices, meta={})
    spec = type("Spec", (), {
        "data": object(), "required_instruments": None, "code_ref": "x:y", "params": {},
        "costs": type("C", (), {"taker_fee_bps": 0.0, "slippage_bps": 0.0})()})()
    import qlab.pipeline.evaluate as evaluate
    import qlab.pipeline.spec as spec_module

    monkeypatch.setattr(spec_module, "load_spec", lambda p: spec)
    monkeypatch.setattr(night, "with_end", lambda s, end: s)
    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, s: panel)
    monkeypatch.setattr(evaluate, "resolve_strategy", lambda ref: Strategy())

    net = decay.stand_net_return(None, "x.yaml", end=date(2026, 4, 10),
                                 cut=pd.Timestamp("2026-02-20", tz="UTC"))

    assert seen_rows == [51]  # 2026-01-01 .. 2026-02-20 inclusive
    assert net.index[-1] < pd.Timestamp("2026-02-20", tz="UTC")
    with pytest.raises(decay.StandHistoryMissing):
        decay.stand_net_return(None, "x.yaml", end=date(2026, 4, 10),
                               cut=pd.Timestamp("2026-01-02", tz="UTC"))
    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, s: None)
    with pytest.raises(decay.StandHistoryMissing, match="not cached"):
        decay.stand_net_return(None, "x.yaml", end=date(2026, 4, 10),
                               cut=pd.Timestamp("2026-02-20", tz="UTC"))
