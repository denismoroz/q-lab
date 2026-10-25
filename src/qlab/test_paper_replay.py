"""Tests for `qlab.paper_replay` and the replay scripts of the re-tuned paper books."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qlab import paper_replay as pr
from qlab.paper_rule import MIN_REPLAY_DAYS, ReplayFigures, parse_replay_lines

HOUR_MS = pr.HOUR_MS
START = pd.Timestamp("2026-10-01 14:00", tz="UTC")


def _curve(hours: int, *, daily_growth: float = 0.001, start: pd.Timestamp = START,
           wiggle: float = 0.0) -> pd.Series:
    """An hourly equity curve; `wiggle` adds a deterministic day-to-day variation."""
    index = pd.date_range(start, periods=hours, freq="h")
    days = np.arange(hours) / 24.0
    noise = wiggle * np.sin(np.floor(days) * 1.7)
    return pd.Series(1000.0 * (1 + daily_growth * days + noise), index=index)


# ── daily returns and the comparison ────────────────────────────────────────

def test_daily_returns_are_midnight_to_midnight_and_skip_broken_days() -> None:
    index = pd.date_range("2026-10-01 00:00", periods=24 * 5 + 1, freq="h", tz="UTC")
    curve = pd.Series(np.arange(len(index), dtype=float) + 100.0, index=index)
    curve = curve.drop(pd.Timestamp("2026-10-03 00:00", tz="UTC"))  # a missing midnight

    returns = pr.daily_returns(curve)

    # 1->2, 4->5 and 5->6 Oct have both midnights one day apart; 2->4 spans the hole
    assert list(returns.index.strftime("%m-%d")) == ["10-02", "10-05", "10-06"]
    assert returns.iloc[0] == pytest.approx(124.0 / 100.0 - 1)
    assert returns.iloc[1] == pytest.approx(196.0 / 172.0 - 1)


def test_a_young_live_part_has_no_correlation() -> None:
    paper = _curve(60, wiggle=0.01)  # two and a half days
    replay = paper.copy()

    result = pr.compare_curves(paper, replay)

    assert result.hours == 60
    assert result.days == 2
    assert result.correlation is None
    assert result.max_abs_diff == 0.0


def test_an_identical_old_enough_live_part_correlates_fully() -> None:
    paper = _curve(24 * (MIN_REPLAY_DAYS + 3), wiggle=0.01)
    replay = paper * 1.0001

    result = pr.compare_curves(paper, replay)

    assert result.days >= MIN_REPLAY_DAYS
    assert result.correlation == pytest.approx(1.0, abs=1e-6)
    assert result.paper_return == pytest.approx(result.replay_return, abs=2e-4)


def test_a_replay_without_variation_has_no_correlation_not_a_nan_figure() -> None:
    paper = _curve(24 * 40, wiggle=0.01)
    flat = pd.Series(1000.0, index=paper.index)

    result = pr.compare_curves(paper, flat)

    assert result.days >= MIN_REPLAY_DAYS
    assert result.correlation is None


def test_curves_that_share_no_hour_compare_as_empty() -> None:
    a = _curve(10)
    b = _curve(10, start=START + pd.Timedelta(days=3))

    result = pr.compare_curves(a, b)

    assert result.hours == 0 and result.correlation is None and result.paper_return is None


def test_cut_live_keeps_the_launch_row_as_the_base() -> None:
    curve = _curve(10)
    live = pr.cut_live(curve, START + pd.Timedelta(hours=4))
    assert live.index[0] == START + pd.Timedelta(hours=4) and len(live) == 6


# ── what the rule reads: an undefined correlation is never a match ──────────

def _block(c: pr.Comparison, **kw) -> list[str]:
    return ["**trend_retune_3k**", *pr.comparison_lines(c, **kw)]


def test_no_live_hours_prints_no_figure_and_is_not_a_match() -> None:
    lines = _block(pr.compare_curves(_curve(1), _curve(1)), prices=False, money=True)

    figures = parse_replay_lines(lines)["trend_retune_3k"]

    assert any("not computed" in ln for ln in lines)
    assert figures.correlation is None and figures.paper is None
    assert figures.matches() is None


def test_two_days_of_live_hours_say_so_plainly_and_are_not_a_match() -> None:
    curve = _curve(60, wiggle=0.01)
    lines = ["book 8 (b2_retune)", *pr.comparison_lines(pr.compare_curves(curve, curve))]

    figures = parse_replay_lines(lines)["book 8"]

    assert ("daily return correlation not computed: n=2 complete UTC days, "
            f"at least {MIN_REPLAY_DAYS} needed") in \
        [ln.strip() for ln in lines]
    assert figures.paper is not None and figures.replay is not None  # the totals are shown
    assert figures.correlation is None
    assert figures.matches() is None


def test_a_printed_correlation_on_too_few_days_never_reads_as_a_match() -> None:
    # Whatever a script prints, the rule itself refuses a figure resting on a few days.
    lines = ["book 8 (b2_retune)", "  paper  +1.00%   replay +1.00%",
             "  daily return correlation 1.0000 (n=2)"]

    figures = parse_replay_lines(lines)["book 8"]

    assert figures.correlation == 1.0 and figures.days == 2
    assert figures.matches() is None
    assert ReplayFigures(correlation=1.0, days=MIN_REPLAY_DAYS - 1).matches() is None
    assert ReplayFigures(correlation=0.5, days=MIN_REPLAY_DAYS - 1).matches() is None
    assert ReplayFigures(correlation=0.95, days=MIN_REPLAY_DAYS).matches() is True
    assert ReplayFigures(correlation=0.5, days=MIN_REPLAY_DAYS).matches() is False
    # An older script's block without the day count keeps its old reading.
    assert ReplayFigures(correlation=0.95).matches() is True


def test_an_old_enough_block_is_read_as_before() -> None:
    curve = _curve(24 * 40, wiggle=0.01)
    lines = _block(pr.compare_curves(curve, curve * 1.00001), prices=False, money=True)

    figures = parse_replay_lines(lines)["trend_retune_3k"]

    assert figures.days is not None and figures.days >= MIN_REPLAY_DAYS
    assert figures.matches() is True


# ── recorded data and daily candles ─────────────────────────────────────────

def _write_recorded(root: Path, coin: str, hours: int, *, start: str = "2026-10-01 00:00",
                    skip: set[int] | None = None) -> pd.DatetimeIndex:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="timestamp")
    keep = [i for i in range(hours) if i not in (skip or set())]
    candles = pd.DataFrame(
        {"open": 1.0, "high": [2.0 + i for i in keep], "low": 0.5,
         "close": [1.0 + i for i in keep], "volume": 10.0, "trade_count": 1},
        index=index[keep])
    (root / "candles" / "1h").mkdir(parents=True, exist_ok=True)
    candles.to_parquet(root / "candles" / "1h" / f"{coin}.parquet")
    stamps = index + pd.Timedelta(milliseconds=45)  # settlements are stamped a few ms late
    funding = pd.DataFrame({"funding_rate": 0.0001}, index=stamps.rename("timestamp"))
    (root / "funding").mkdir(parents=True, exist_ok=True)
    funding.to_parquet(root / "funding" / f"{coin}.parquet")
    return index


def test_recorded_hourly_is_keyed_by_the_hour_and_funding_is_floored(tmp_path) -> None:
    index = _write_recorded(tmp_path, "BTC", 6)

    rec = pr.read_recorded_hourly(tmp_path, "BTC")

    first = pr.ms(index[0])
    assert rec.close[first] == 1.0 and rec.close[first + 5 * HOUR_MS] == 6.0
    assert rec.high[first] == 2.0
    assert set(rec.funding) == {first + i * HOUR_MS for i in range(6)}  # floored from +45 ms
    assert rec.funding_records[0] == (first + 45, 0.0001)


def test_on_grid_marks_the_holes_as_nan() -> None:
    values = {0: 1.0, 2 * HOUR_MS: 3.0, 9 * HOUR_MS: 99.0}  # the last is past the grid

    grid = pr.on_grid(values, 0, 3 * HOUR_MS)

    assert grid[0] == 1.0 and grid[2] == 3.0
    assert np.isnan(grid[1]) and np.isnan(grid[3]) and len(grid) == 4


def test_daily_candles_from_hourly_ones_need_all_24_hours(tmp_path) -> None:
    _write_recorded(tmp_path, "BTC", 24 * 3, skip={30})  # an hour missing on the second day
    rec = pr.read_recorded_hourly(tmp_path, "BTC")
    day0 = pr.ms(pd.Timestamp("2026-10-01", tz="UTC"))

    daily = pr.daily_from_hourly(rec, day0, day0 + 2 * pr.DAY_MS)

    assert set(daily) == {day0, day0 + 2 * pr.DAY_MS}
    assert daily[day0] == (24.0, 240.0)  # the last hour's close, the sum of the volumes


def test_daily_candles_and_funding_from_a_raw_cache_frame() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-10-01 00:00:00.006", tz="UTC"),
         pd.Timestamp("2026-10-02", tz="UTC")])
    frame = pd.DataFrame({"price": [10.0, np.nan, 11.0], "funding": [np.nan, 0.0002, np.nan],
                          "volume": [5.0, np.nan, np.nan]}, index=index)

    daily = pr.daily_from_raw(frame)
    funding = pr.funding_from_raw(frame)

    day0 = pr.ms(index[0])
    assert daily == {day0: (10.0, 5.0), day0 + pr.DAY_MS: (11.0, None)}
    assert funding == [(day0 + 6, 0.0002)]


def test_the_stand_cache_wins_where_both_sources_have_a_day() -> None:
    assert pr.extend_daily({1: (1.0, 1.0)}, {1: (9.0, 9.0), 2: (2.0, 2.0)}) == {
        1: (1.0, 1.0), 2: (2.0, 2.0)}
    assert pr.merge_funding([(2, 0.2), (1, 0.1)], [(2, 9.0), (3, 0.3)]) == [
        (1, 0.1), (2, 0.2), (3, 0.3)]


def test_the_newest_raw_file_of_a_coin_is_found(tmp_path) -> None:
    for name in ("BTC__2025-01-01__2026-10-01.parquet", "BTC__2025-01-01__2026-10-24.parquet",
                 "BTC-SPOT__2025-01-01__2026-09-20.parquet"):
        (tmp_path / name).write_bytes(b"")
    assert pr.latest_raw_file(tmp_path, "BTC").name == "BTC__2025-01-01__2026-10-24.parquet"
    assert pr.latest_raw_file(tmp_path, "ETH") is None


# ── the Bv2 retune script, end to end on a synthetic store ──────────────────

def _script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retune_params(coins: list[str]) -> dict:
    return {
        "coins": coins, "capital_usd": 257.0, "spot_share": 0.5, "hedge_threshold": 0.0,
        "sticky_exit_hours": 12, "ratchet_threshold": 0.5, "carry_enabled": False,
        "carry_fraction": 0.6, "carry_entry_apr": 0.1, "carry_exit_hours": 24, "slippage": 0.0005,
        "min_order_usd": 10.0, "margin_enabled": True,
        "short_leverage": {c: 2.0 for c in coins}, "default_leverage": 1.0,
        "maint_margin_rate": {c: 0.02 for c in coins}, "default_maint_margin_rate": 0.05,
        "margin_buffer": 0.1, "hedge_margin_headroom": 1.0, "rebalance_at_im_share": 0.5,
        "mode": "paper",
        "retune_candidates": {"b2": {"hedge_margin_headroom": 1.0},
                              "b2_cold": {"hedge_margin_headroom": 1.5}},
        "retune_history_start": "2026-01-01T00:00:00Z",
        "retune_coin_starts": {c: "2026-02-03T20:00:00Z" for c in coins},  # 788 hours after
        "retune_refit_every": "M",
    }


@pytest.fixture(scope="module")
def b2_world(tmp_path_factory):
    """Two coins on 3000 synthetic hourly candles from 2026-01-01 (first monthly choice on
    April 1st), the book's params, and the paper side: the replay's own rows, which a
    replay of unchanged code must reproduce hour by hour."""
    root = tmp_path_factory.mktemp("b2_retune")
    record = root / "recorded"
    coins = ["BTC", "ETH"]
    rng = np.random.default_rng(7)
    hours = 3000
    index = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC", name="timestamp")
    for coin in coins:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, hours)))
        candles = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                                "close": close, "volume": 5.0, "trade_count": 1}, index=index)
        (record / "candles" / "1h").mkdir(parents=True, exist_ok=True)
        candles.to_parquet(record / "candles" / "1h" / f"{coin}.parquet")
        funding = pd.DataFrame({"funding_rate": rng.normal(0.00002, 0.00002, hours)},
                               index=(index + pd.Timedelta(milliseconds=30)).rename("timestamp"))
        (record / "funding").mkdir(parents=True, exist_ok=True)
        funding.to_parquet(record / "funding" / f"{coin}.parquet")
    params_path = root / "params.json"
    params_path.write_text(json.dumps(_retune_params(coins)))

    script = _script("replay_b2_retune_paper.py")
    params = script.retune_params.B2RetuneParams.from_dict(json.loads(params_path.read_text()))
    last_bar = int(index[-1].timestamp() * 1000)
    series = script.build_series(params, record, last_bar)
    replay, choices, last = script.run_replay(params, series, last_bar)
    paper = replay.assign(strategy_id=8)[["strategy_id", "ts_ms", "coin", "price", "equity"]]
    paper_path = root / "b2_equity.csv"
    paper.to_csv(paper_path, index=False)
    events = pd.DataFrame({"ts_ms": [e["bar_ms"] + HOUR_MS for e in choices],
                           "details_json": [json.dumps({"refit": e["refit"], "choice": e["choice"],
                                                         "scores": e["scores"]}) for e in choices]})
    events_path = root / "events.csv"
    events.to_csv(events_path, index=False)
    return {"script": script, "record": record, "params": params_path, "paper": paper_path,
            "events": events_path, "choices": choices, "replay": replay, "last": last,
            "root": root}


def test_the_synthetic_book_makes_a_first_choice_and_publishes_per_coin_rows(b2_world) -> None:
    replay, choices = b2_world["replay"], b2_world["choices"]

    assert {e["choice"] for e in choices if e["refit"] >= 3} <= {"b2", "b2_cold"}
    assert any(e["choice"] for e in choices)  # the book is born
    assert set(replay.coin) == {"BTC", "ETH"}
    assert replay.groupby("coin").size().nunique() == 1


def _run(world, monkeypatch, capsys, live_from: str, *extra: str) -> list[str]:
    argv = ["x", "--paper", str(world["paper"]), "--params", str(world["params"]),
            "--events", str(world["events"]), "--live-from", live_from,
            "--record-dir", str(world["record"]), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    world["script"].main()
    return capsys.readouterr().out.splitlines()


def test_an_unchanged_replay_matches_the_paper_rows_hour_by_hour(b2_world, monkeypatch,
                                                                  capsys) -> None:
    lines = _run(b2_world, monkeypatch, capsys, "2026-04-02T00:00:00+00:00")

    figures = parse_replay_lines(lines)["book 8"]
    assert "price mismatches 0" in " ".join(lines)
    assert "max |hourly equity diff| 0.0000" in " ".join(lines)
    assert figures.correlation == pytest.approx(1.0, abs=1e-9)
    assert figures.days >= MIN_REPLAY_DAYS
    assert figures.matches() is True
    assert any(ln.strip().startswith("history re-tunings") and "same choice" in ln for ln in lines)


def test_a_launch_a_day_before_the_end_is_not_judged(b2_world, monkeypatch, capsys) -> None:
    lines = _run(b2_world, monkeypatch, capsys, "2026-05-04T12:00:00+00:00")

    figures = parse_replay_lines(lines)["book 8"]
    assert figures.correlation is None
    assert figures.matches() is None
    assert any("not computed" in ln for ln in lines)


def test_a_launch_at_the_last_row_has_no_live_hour_yet(b2_world, monkeypatch, capsys) -> None:
    last = pd.Timestamp(b2_world["last"] + HOUR_MS, unit="ms", tz="UTC").isoformat()

    lines = _run(b2_world, monkeypatch, capsys, last)

    figures = parse_replay_lines(lines)["book 8"]
    assert "matched hours 1," in " ".join(lines)
    assert figures.paper is None and figures.correlation is None and figures.matches() is None


def test_a_replay_that_disagrees_with_the_paper_shows_it(b2_world, monkeypatch, capsys,
                                                         tmp_path) -> None:
    paper = pd.read_csv(b2_world["paper"])
    paper.loc[paper.ts_ms > paper.ts_ms.quantile(0.9), "equity"] *= 1.05  # the paper drifted away
    drifted = tmp_path / "b2_equity.csv"
    paper.to_csv(drifted, index=False)
    world = {**b2_world, "paper": drifted}

    lines = _run(world, monkeypatch, capsys, "2026-04-02T00:00:00+00:00")

    figures = parse_replay_lines(lines)["book 8"]
    live_diff = next(ln.strip() for ln in lines
                     if ln.strip().startswith("max |hourly equity diff|"))
    assert live_diff != "max |hourly equity diff| 0.0000"
    assert figures.paper != pytest.approx(figures.replay, abs=1e-3)


def test_the_replay_file_has_the_columns_the_paper_check_reads(b2_world, monkeypatch, capsys,
                                                                tmp_path) -> None:
    out = tmp_path / "replay_b2_retune.csv"

    _run(b2_world, monkeypatch, capsys, "2026-04-02T00:00:00+00:00", "--out", str(out))

    frame = pd.read_csv(out)
    assert list(frame.columns) == ["strategy_id", "ts_ms", "equity", "replay_equity"]
    assert set(frame.strategy_id) == {8}
    assert (frame.equity - frame.replay_equity).abs().max() == pytest.approx(0.0, abs=1e-9)


# ── the trend retune script's own helpers ───────────────────────────────────

def test_the_logged_universes_are_found_by_decision_bar() -> None:
    script = _script("replay_trend_retune_paper.py")
    bar = pr.ms(pd.Timestamp("2026-10-26", tz="UTC"))
    events = pd.DataFrame([
        {"ts_ms": bar + HOUR_MS, "kind": "retune_day",
         "details_json": json.dumps({"universes": {"cap": ["BTC"], "volume": ["ETH"]}})},
        {"ts_ms": bar + HOUR_MS, "kind": "open", "details_json": "{}"},
    ])

    assert script.logged_universes(events) == {bar: {"cap": ["BTC"], "volume": ["ETH"]}}


def test_listed_perps_leave_out_the_delisted_ones(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "2026-10-24.json").write_text(json.dumps({"universe": [{"name": "OLD"}]}))
    (meta / "2026-10-25.json").write_text(json.dumps({"universe": [
        {"name": "BTC"}, {"name": "GONE", "isDelisted": True},
        {"name": "ETH", "isDelisted": False}]}))

    assert _script("replay_trend_retune_paper.py").listed_perps(tmp_path) == {"BTC", "ETH"}


def test_the_history_check_counts_choices_and_scores() -> None:
    script = _script("replay_trend_retune_paper.py")
    ours = [{"week": "2026-W01", "choice": "cap/ens", "scores": {"cap/ens": 0.5, "cap/l30": None}},
            {"week": "2026-W02", "choice": "cap/l30", "scores": {"cap/ens": 0.1, "cap/l30": 0.2}}]
    events = pd.DataFrame([
        {"kind": "retune", "details_json": json.dumps(
            {"week": "2026-W01", "choice": "cap/ens", "phase": "history",
             "scores": {"cap/ens": 0.5, "cap/l30": None}})},
        {"kind": "retune", "details_json": json.dumps(
            {"week": "2026-W02", "choice": "cap/ens", "phase": "history",
             "scores": {"cap/ens": 0.1, "cap/l30": 0.25}})},
        {"kind": "retune", "details_json": json.dumps(
            {"week": "2026-W03", "choice": "cap/ens", "phase": "live", "scores": {}})},
        {"kind": "open", "details_json": "{}"},
    ])
    result = SimpleNamespace(events=ours, state=SimpleNamespace(chosen="cap/l30"))

    (line,) = script.check_history(result, events)

    assert "frab 2 weekly re-tunings, replay 2, same choice 1" in line
    assert "scores off by more than 1e-06: 1" in line


# ── the night's wiring ──────────────────────────────────────────────────────

def test_the_night_replays_the_retuned_books_from_the_mapping(tmp_path, monkeypatch) -> None:
    from qlab import night, paper_check

    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n"
        "  bv2-research-headroom-retune:\n    book: b2_retune\n    paper_file: b2_equity.csv\n"
        "    strategy_capital_usd: 257\n"
        "    strategy_id: 8\n    sum_coins: true\n    replay_file: replay_b2_retune.csv\n"
        "    replay_block: 'book 8'\n    realised_from: 2026-10-25T14:00:00+00:00\n"
        "  trend-retune-weekly:\n    book: trend_retune_3k\n"
        "    paper_file: trend_retune_equity.csv\n    strategy_capital_usd: 3000\n"
        "    strategy_id: 6\n    replay_file: replay_trend_retune.csv\n"
        "    replay_block: trend_retune_3k\n")
    monkeypatch.setattr(paper_check, "BOOKS", books)
    asked = []
    monkeypatch.setattr(night, "_ssh_query", lambda sql, out, header=True: asked.append((sql, out)))

    lines, runs = night._retune_replay_runs(tmp_path)

    assert lines == []
    assert set(runs) == {"Bv2 re-tuned monthly (book 8 = b2_retune)", "trend_retune_3k"}
    b2 = runs["Bv2 re-tuned monthly (book 8 = b2_retune)"]
    assert b2[0] == "scripts/replay_b2_retune_paper.py"
    assert b2[b2.index("--live-from") + 1] == "2026-10-25T14:00:00+00:00"
    assert b2[b2.index("--out") + 1] == str(tmp_path / "replay_b2_retune.csv")
    trend = runs["trend_retune_3k"]
    assert trend[0] == "scripts/replay_trend_retune_paper.py"
    assert trend[trend.index("--strategy-id") + 1] == "6"
    # the exports are SELECTs on the book's own rows, and nothing else
    assert all(sql.lower().startswith("select") for sql, _ in asked)
    assert {out.name for _, out in asked} == {
        "b2_retune_params.json", "b2_retune_events.csv", "trend_retune_params.json",
        "trend_retune_events.csv"}
    # the label of a trend block starts with the block name the rule looks for
    assert parse_replay_lines(["**trend_retune_3k**",
                               "    daily return correlation 0.9500 (n=30)"])[
        "trend_retune_3k"].matches() is True


def test_a_failing_export_costs_only_that_books_replay(tmp_path, monkeypatch) -> None:
    import subprocess

    from qlab import night, paper_check

    books = tmp_path / "books.yaml"
    books.write_text(
        "books:\n  trend-retune-weekly:\n    book: trend_retune_3k\n    strategy_id: 6\n"
        "    strategy_capital_usd: 3000\n"
        "    replay_block: trend_retune_3k\n")
    monkeypatch.setattr(paper_check, "BOOKS", books)

    def broken(sql, out, header=True):
        raise subprocess.CalledProcessError(1, "ssh")

    monkeypatch.setattr(night, "_ssh_query", broken)

    lines, runs = night._retune_replay_runs(tmp_path)

    assert runs == {}
    assert len(lines) == 1 and lines[0].startswith("trend retune book not replayed")


def test_a_mapping_without_the_retuned_books_adds_no_replay(tmp_path, monkeypatch) -> None:
    from qlab import night, paper_check

    books = tmp_path / "books.yaml"
    books.write_text("books:\n  other:\n    book: b2\n    strategy_id: 3\n"
                     "    strategy_capital_usd: 257\n")
    monkeypatch.setattr(paper_check, "BOOKS", books)
    monkeypatch.setattr(night, "_ssh_query", lambda *a, **kw: pytest.fail("nothing to export"))

    assert night._retune_replay_runs(tmp_path) == ([], {})


def test_the_shipped_mapping_names_the_retuned_books_replays() -> None:
    from qlab import paper_check

    root = Path(__file__).resolve().parents[2]
    books = paper_check.load_books(root / "night" / "paper_books.yaml")

    b2, trend = books["bv2-research-headroom-retune"], books["trend-retune-weekly"]
    assert (b2["replay_block"], b2["replay_file"]) == ("book 8", "replay_b2_retune.csv")
    assert (trend["replay_block"], trend["replay_file"]) == (
        "trend_retune_3k", "replay_trend_retune.csv")
    for script in ("replay_b2_retune_paper.py", "replay_trend_retune_paper.py"):
        assert (root / "scripts" / script).is_file()
