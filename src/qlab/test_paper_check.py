"""Tests for `qlab.paper_check`: the paper books against the stand over one window."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab import night, paper_check
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    PaperCheck,
    Profile,
    SourceType,
    TrialRoute,
    TrialStatus,
    VerdictStage,
)

HOUR_MS = 3_600_000


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _b2_csv(path: Path, books: dict[int, list[tuple[int, dict[str, float]]]]) -> None:
    """books: strategy_id -> [(ts_ms, {coin: equity})]."""
    rows = [{"strategy_id": sid, "ts_ms": ts, "coin": coin, "price": 1.0, "equity": eq}
            for sid, series in books.items() for ts, coins in series for coin, eq in coins.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_b2_book_is_the_sum_of_its_coins_over_exactly_the_window(tmp_path) -> None:
    start = date(2026, 10, 3)
    series = [(_ms(start) + h * HOUR_MS, {"BTC": 100.0 + h, "ETH": 50.0}) for h in range(0, 49)]
    # The last timestamp has only one coin: it must be dropped, not summed short.
    series.append((_ms(start) + 49 * HOUR_MS, {"BTC": 1000.0}))
    other_book = [(ts, {"BTC": 1.0, "ETH": 1.0}) for ts, _ in series[:-1]]
    csv = tmp_path / "b2_equity.csv"
    _b2_csv(csv, {4: series, 3: other_book})

    curve = paper_check.equity_curve(csv, strategy_id=4, sum_coins=True)
    assert len(curve) == 49 and curve.iloc[0] == 150.0

    # One day: rows from 10-03 00:00 up to and including 10-04 00:00.
    day = paper_check.window_return(curve, datetime(2026, 10, 3, tzinfo=UTC),
                                    datetime(2026, 10, 4, tzinfo=UTC))
    assert day is not None and day.rows == 25
    assert day.total == pytest.approx(174.0 / 150.0 - 1)
    assert day.first == datetime(2026, 10, 3, tzinfo=UTC)
    assert day.last == datetime(2026, 10, 4, tzinfo=UTC)

    # A start before the book began: the first row at or after it.
    late = paper_check.window_return(curve, datetime(2026, 9, 1, tzinfo=UTC), None)
    assert late is not None and late.first == datetime(2026, 10, 3, tzinfo=UTC)
    # Fewer than two rows in the window: no number.
    assert paper_check.window_return(curve, datetime(2026, 12, 1, tzinfo=UTC), None) is None


def test_the_trend_book_is_one_row_per_timestamp(tmp_path) -> None:
    csv = tmp_path / "trend_equity.csv"
    ts = [_ms(date(2026, 10, 1)) + h * HOUR_MS for h in range(5)]
    pd.DataFrame({"strategy_id": 5, "ts_ms": ts,
                  "equity": [3000.0, 3030.0, 2970.0, 3060.0, 3300.0]}).to_csv(csv, index=False)

    curve = paper_check.equity_curve(csv, strategy_id=5)
    part = paper_check.window_return(curve, datetime(2026, 10, 1, 1, tzinfo=UTC),
                                     datetime(2026, 10, 1, 3, tzinfo=UTC))

    assert part is not None and part.rows == 3
    assert part.total == pytest.approx(3060.0 / 3030.0 - 1)


def test_the_replay_is_read_on_the_rows_the_paper_window_used(tmp_path) -> None:
    csv = tmp_path / "replay.csv"
    ts = [_ms(date(2026, 10, 1)) + h * HOUR_MS for h in range(4)]
    pd.DataFrame({"strategy_id": 4, "ts_ms": ts, "equity": [1.0] * 4,
                  "replay_equity": [200.0, 210.0, 190.0, 220.0]}).to_csv(csv, index=False)
    a, b = (datetime.fromtimestamp(ts[i] / 1000, tz=UTC) for i in (1, 3))

    assert paper_check.replay_window_return(csv, a, b, strategy_id=4) == pytest.approx(
        220 / 210 - 1)
    # A window end the replay does not have: unknown, not guessed.
    assert paper_check.replay_window_return(csv, a, b + timedelta(hours=5), strategy_id=4) is None
    assert paper_check.replay_window_return(csv, a, b, strategy_id=3) is None


def test_the_forward_total_is_the_yearly_figure_over_its_days() -> None:
    assert paper_check.forward_total(-0.34, 21.0) == pytest.approx(0.66 ** (21 / 365.25) - 1)
    assert paper_check.forward_total(0.5, 365.25) == pytest.approx(0.5)
    assert paper_check.forward_total(-1.5, 10.0) == -1.0


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _stand(session, idea_id="bv2-x", *, ann=-0.34, days=505 / 24, forward=1.0,
           start=date(2026, 10, 3), end=date(2026, 10, 24), extra=None):
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                         rebalance="1h", costs_model={}, code_ref="a:b")
    repo.add_data_snapshot(session, id=f"snap-{idea_id}", source="hyperliquid", instruments={},
                           range_start=date(2025, 1, 1), range_end=end, path="x", rows=1)
    trials = []
    for tag, route in (("old", TrialRoute.NEEDS_FORWARD), ("new", TrialRoute.NEEDS_FORWARD)):
        metrics = {"ann_return_net": ann if tag == "new" else 9.9, "forward_days": days,
                   "judged_on_forward": forward, **((extra or {}) if tag == "new" else {})}
        trial = repo.add_trial(session, spec_id=spec.id, config_hash=tag, code_sha="x", params={},
                               status=TrialStatus.OK, metrics=metrics, route=route,
                               snapshot_id=f"snap-{idea_id}")
        repo.add_verdicts(session, [dict(
            idea_id=idea_id, spec_id=spec.id, trial_id=trial.id, stage=VerdictStage.EDGE,
            rule_id="r", rules_version="v", metric="sharpe_net", value=1.0, comparator=">=",
            threshold=0.5, passed=True, data_range_start=a, data_range_end=b)
            for a, b in ((date(2025, 1, 1), date(2026, 10, 2)), (start, end))])
        trials.append(trial)
    return trials[-1]


def test_the_stand_window_and_forward_total_come_from_the_latest_trial(session) -> None:
    trial = _stand(session)

    stand = paper_check.stand_window(session, "bv2-x", date(2026, 10, 3))

    assert stand is not None and stand.trial_id == trial.id
    assert (stand.start, stand.end) == (date(2026, 10, 3), date(2026, 10, 24))
    assert stand.total == pytest.approx(0.66 ** ((505 / 24) / 365.25) - 1)


def test_a_run_that_judged_no_forward_test_gives_no_stand_window(session) -> None:
    _stand(session, forward=0.0)

    assert paper_check.stand_window(session, "bv2-x", date(2026, 10, 3)) is None
    assert paper_check.stand_window(session, "nobody", date(2026, 10, 3)) is None


def _books(tmp_path: Path) -> dict:
    return {"bv2-x": {"book": "b2_cold", "spec": "x.yaml", "label": "Bv2, запас 1.5",
                      "paper_file": "b2_equity.csv", "strategy_id": 4, "sum_coins": True,
                      "replay_file": "replay_b2.csv"}}


def _fixed(monkeypatch, day: date) -> None:
    import qlab.pipeline.spec as spec_module

    monkeypatch.setattr(spec_module, "load_spec", lambda p: SimpleNamespace(params_fixed_at=day))


def _paper_fixture(folder: Path, first: date, hours: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    ts = [_ms(first) + h * HOUR_MS for h in range(hours)]
    series = [(t, {"BTC": 100.0 - 0.001 * i, "ETH": 100.0}) for i, t in enumerate(ts)]
    _b2_csv(folder / "b2_equity.csv", {4: series})
    pd.DataFrame({"strategy_id": 4, "ts_ms": ts, "equity": 1.0,
                  "replay_equity": [200.0 - 0.001 * i for i in range(hours)]}
                 ).to_csv(folder / "replay_b2.csv", index=False)


def test_a_check_writes_one_row_per_night_and_a_plain_line(session, tmp_path, monkeypatch) -> None:
    _stand(session)
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "2026-10-25"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)
    day = date(2026, 10, 25)

    lines = paper_check.run_paper_checks(session, day, folder, _books(tmp_path))
    again = paper_check.run_paper_checks(session, day, folder, _books(tmp_path))

    rows = session.query(PaperCheck).all()
    assert len(rows) == 1 and lines == again
    row = rows[0]
    assert (row.window_start, row.window_end) == (date(2026, 10, 3), date(2026, 10, 24))
    assert row.stand_trial_id is not None and row.note == ""
    assert row.stand_return == pytest.approx(0.66 ** ((505 / 24) / 365.25) - 1)
    # 505 hours after 10-03 00:00 (21 days and the last hourly bar): both end rows exist.
    first = 20 * 24 + 0
    assert row.paper_return == pytest.approx(
        (200.0 - 0.001 * (first + 505)) / (200.0 - 0.001 * first) - 1)
    assert row.replay_return is not None
    assert lines[0].startswith("Bv2, запас 1.5: за 03.10–24.10 бумага −")
    assert ", стенд −2.4%" in lines[0] or ", стенд −2." in lines[0]


def test_windows_that_differ_are_said_to_differ(session, tmp_path, monkeypatch) -> None:
    _stand(session)
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "p"
    _paper_fixture(folder, date(2026, 10, 10), 24 * 20)  # the book began a week after the window

    lines = paper_check.run_paper_checks(session, date(2026, 10, 25), folder, _books(tmp_path))

    row = session.query(PaperCheck).one()
    assert row.window_start == date(2026, 10, 10)
    assert (row.stand_window_start, row.stand_window_end) == (date(2026, 10, 3), date(2026, 10, 24))
    assert row.note.startswith("Отрезки разные")
    assert "числа сравнивать нельзя" in lines[0]


def test_what_is_missing_is_a_line_not_a_crash(session, tmp_path, monkeypatch) -> None:
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "empty"
    folder.mkdir()
    day = date(2026, 10, 25)

    no_idea = paper_check.run_paper_checks(session, day, folder, _books(tmp_path))
    _stand(session)
    no_file = paper_check.run_paper_checks(session, day, folder, _books(tmp_path))
    _paper_fixture(folder, date(2026, 9, 1), 24)  # a day of rows, none inside the window
    no_rows = paper_check.run_paper_checks(session, day, folder, _books(tmp_path))

    assert "нет в реестре" in no_idea[0]
    assert "нет выгрузки" in no_file[0]
    assert "нет двух точек" in no_rows[0]
    assert session.query(PaperCheck).count() == 0


def test_rows_before_realised_from_are_not_paper(session, tmp_path, monkeypatch) -> None:
    # A book that replayed history at launch (frab's b2_retune): the replayed rows are the
    # stand's own arithmetic, so the paper window starts at the launch, not at the stand's.
    _stand(session)
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "p"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)
    books = _books(tmp_path)
    books["bv2-x"]["realised_from"] = "2026-10-10T00:00:00+00:00"

    lines = paper_check.run_paper_checks(session, date(2026, 10, 25), folder, books)

    row = session.query(PaperCheck).one()
    assert row.window_start == date(2026, 10, 10)
    assert row.note.startswith("Отрезки разные")
    assert "числа сравнивать нельзя" in lines[0]


def test_paper_that_began_after_the_stand_forward_test_has_no_common_window(
        session, tmp_path, monkeypatch) -> None:
    _stand(session)  # forward test 10-03 .. 10-24
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "p"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)
    books = _books(tmp_path)
    books["bv2-x"]["realised_from"] = "2026-10-25T14:00:00+00:00"

    lines = paper_check.run_paper_checks(session, date(2026, 10, 25), folder, books)

    assert "общего отрезка пока нет" in lines[0]
    assert session.query(PaperCheck).count() == 0


def test_without_a_stand_forward_test_the_paper_side_is_still_recorded(
        session, tmp_path, monkeypatch) -> None:
    _stand(session, forward=0.0)
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "p"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)

    paper_check.run_paper_checks(session, date(2026, 10, 25), folder, _books(tmp_path))

    row = session.query(PaperCheck).one()
    assert row.stand_return is None and row.stand_trial_id is None
    assert "нет проверки вперёд" in row.note


def test_the_night_step_exports_replays_and_writes_the_check(tmp_path, monkeypatch) -> None:
    import qlab.registry.db as db_module
    from qlab.registry.db import session_scope

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "q.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    monkeypatch.setattr(night, "PAPER_DIR", tmp_path / "paper")
    day = date(2026, 10, 25)
    with session_scope() as s:
        _stand(s)
    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n  bv2-x:\n    book: b2_cold\n    spec: x.yaml\n    label: 'Bv2, запас 1.5'\n"
        "    paper_file: b2_equity.csv\n    strategy_id: 4\n    sum_coins: true\n"
        "    strategy_capital_usd: 257\n    replay_file: replay_b2.csv\n")
    monkeypatch.setattr(paper_check, "BOOKS", books)
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "paper" / day.isoformat()

    def fake_export(sql, out, header=True):
        out.parent.mkdir(parents=True, exist_ok=True)
        if "from b2_equity" in sql:
            _paper_fixture(folder, date(2026, 9, 13), 24 * 45)  # also leaves a replay file
        else:
            out.write_text("")

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="paper +1.00%   replay +1.00%\n",
                                           stderr="")

    monkeypatch.setattr(night, "_ssh_query", fake_export)
    monkeypatch.setattr(night.subprocess, "run", fake_run)

    lines = night.paper_reconciliation(day)

    assert "**Бумага и стенд за один отрезок**" in lines
    assert any(ln.strip().startswith("Bv2, запас 1.5: за 03.10–24.10") for ln in lines)
    with session_scope() as s:
        assert s.query(PaperCheck).one().idea_id == "bv2-x"
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)


def test_the_shipped_mapping_names_real_specs_and_files() -> None:
    root = Path(__file__).resolve().parents[2]
    books = paper_check.load_books(root / "night" / "paper_books.yaml")

    assert {k: v["book"] for k, v in books.items()} == {
        "strategy-b-v2": "b2", "bv2-research-headroom": "b2_cold", "trend-live": "trend",
        "bv2-research-headroom-retune": "b2_retune", "trend-retune-weekly": "trend_retune_3k"}
    assert {v["strategy_id"] for v in books.values()} == {3, 4, 5, 6, 8}
    # b2_retune replayed its history at launch: only rows from the launch hour are paper.
    assert paper_check.as_utc(books["bv2-research-headroom-retune"]["realised_from"]) == \
        datetime(2026, 10, 25, 14, tzinfo=UTC)
    for cfg in books.values():
        assert (root / cfg["spec"]).is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_shows_a_short_forward_test_as_its_total() -> None:
    web = Path(__file__).resolve().parents[2] / "web" / "src" / "forwardReturn.js"
    script = f"""
    import {{ shortForwardReturn, daysText }} from {str(web.as_uri())!r}
    const m = (judged, days, ann) => ({{
      judged_on_forward: judged, forward_days: days, ann_return_net: ann }})
    const out = {{
      short: shortForwardReturn(m(1, 505 / 24, -0.34)),
      long: shortForwardReturn(m(1, 90, 0.1)),
      selection: shortForwardReturn(m(0, 10, 0.1)),
      none: shortForwardReturn(null),
      wipeout: shortForwardReturn(m(1, 5, -1.2)),
      words: [1, 2, 5, 11, 21, 22, 25].map(daysText),
    }}
    console.log(JSON.stringify(out))
    """
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True,
                          text=True, check=True)
    import json

    out = json.loads(done.stdout)
    assert out["short"]["days"] == 21
    assert out["short"]["total"] == pytest.approx(0.66 ** ((505 / 24) / 365.25) - 1)
    assert out["short"]["yearly"] == -0.34
    assert out["long"] is None and out["selection"] is None and out["none"] is None
    assert out["wipeout"]["total"] == -1
    assert out["words"] == ["1 день", "2 дня", "5 дней", "11 дней", "21 день", "22 дня", "25 дней"]


def test_a_strategy_that_does_not_reinvest_is_compared_in_points_of_its_capital(
        session, tmp_path, monkeypatch) -> None:
    """The stand's trial has its capital summaries (docs/CAPITAL_BASE.md): +20% a year on the
    strategy's money, summed. The paper book (dollars) is read as a gain over the book's
    `strategy_capital_usd`, the stand's total is linear in its days: one base, no compounding."""
    days = 505 / 24
    _stand(session, extra={"cap_ann_return": 0.20, "cap_reinvested": 0.0})
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "2026-10-25"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)
    books = _books(tmp_path)
    books["bv2-x"]["strategy_capital_usd"] = 100

    lines = paper_check.run_paper_checks(session, date(2026, 10, 25), folder, books)

    row = session.query(PaperCheck).one()
    assert row.stand_return == pytest.approx(0.20 * days / 365.25)
    first = 20 * 24
    gain = (100.0 - 0.001 * (first + 505)) - (100.0 - 0.001 * first)  # the BTC leg moved
    assert row.paper_return == pytest.approx(gain / 100.0)
    assert "от денег стратегии" in lines[0]
    assert "docs/CAPITAL_BASE.md" in row.note


def test_a_stand_trial_from_before_the_capital_summaries_is_read_the_old_way(
        session, tmp_path, monkeypatch) -> None:
    _stand(session)  # no cap_ keys
    _fixed(monkeypatch, date(2026, 10, 3))
    folder = tmp_path / "2026-10-25"
    _paper_fixture(folder, date(2026, 9, 13), 24 * 45)
    books = _books(tmp_path)
    books["bv2-x"]["strategy_capital_usd"] = 100

    paper_check.run_paper_checks(session, date(2026, 10, 25), folder, books)

    row = session.query(PaperCheck).one()
    first = 20 * 24
    assert row.paper_return == pytest.approx(
        (200.0 - 0.001 * (first + 505)) / (200.0 - 0.001 * first) - 1), "a ratio, as before"
    assert row.note == ""


def test_the_stand_window_reads_the_capital_figure_when_the_trial_has_it(session) -> None:
    _stand(session, extra={"cap_ann_return": 0.20, "cap_reinvested": 0.0})
    stand = paper_check.stand_window(session, "bv2-x", date(2026, 10, 3))
    assert stand.on_capital and stand.reinvested is False
    assert stand.total == pytest.approx(0.20 * (505 / 24) / 365.25)
    _stand(session, idea_id="trend-x", extra={"cap_ann_return": 0.5, "cap_reinvested": 1.0})
    compounded = paper_check.stand_window(session, "trend-x", date(2026, 10, 3))
    assert compounded.total == pytest.approx(1.5 ** ((505 / 24) / 365.25) - 1)
