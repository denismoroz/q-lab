"""Tests for `qlab.strategy_pnl`: a live strategy's own equity (budget plus its result), never
the exchange account's total. Owner, 2026-10-25: «стратегия работает на 90 ... 43 - это то что
еще доступно на счету - оно не должно никак фигурировать в расчетах»."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qlab import decay, paper_check, strategy_pnl
from qlab.strategy_pnl import HOUR_MS, PnlInputs, StrategyPnlError

T0 = int(datetime(2026, 10, 10, tzinfo=UTC).timestamp() * 1000)


def _leg(position_id: int, farb_id: int, coin: str, instrument: str, *, state: str = "OPEN"):
    side = "LONG" if instrument == "SPOT" else "SHORT" if instrument == "PERP" else "NONE"
    return {"position_id": position_id, "farb_position_id": farb_id, "exchange_id": 1,
            "coin": coin, "instrument": instrument, "side": side, "qty": 1.0,
            "entry_price": 100.0, "opened_at": T0 - HOUR_MS, "closed_at": None,
            "status": "OPEN", "farb_state": state}


def _marks(coin: str, prices: list[float], *, spot: list[float | None] | None = None,
           first_hour: int = -3) -> list[dict]:
    """The freshest row of each clock hour: hour h's row sits at h + 59 minutes, so it is the
    price the boundary h + 1 sees."""
    rows = []
    for i, price in enumerate(prices):
        rows.append({"exchange_id": 1, "coin": coin,
                     "ts_ms": T0 + (first_hour + i) * HOUR_MS + 59 * 60_000, "mark": price,
                     "spot": None if spot is None else spot[i]})
    return rows


def _inputs(*, legs, fills, funding=(), marks, budget: float = 90.0) -> PnlInputs:
    return PnlInputs(
        legs=pd.DataFrame(legs),
        fills=pd.DataFrame(fills, columns=["fill_id", "position_id", "ts_ms", "side", "qty",
                                           "price", "fee"]),
        funding=pd.DataFrame(funding, columns=["position_id", "ts_ms", "amount"]),
        marks=pd.DataFrame(marks, columns=["exchange_id", "coin", "ts_ms", "mark", "spot"]),
        budget=budget)


def _pair_fills(qty: float = 1.0, price: float = 100.0, fee: float = 0.0) -> list[tuple]:
    """A spot long (position 1) and a perp short (position 2) opened before the period."""
    return [(1, 1, T0 - HOUR_MS, "long", qty, price, fee),
            (2, 2, T0 - HOUR_MS, "short", qty, price, fee)]


def _pair_legs() -> list[dict]:
    return [_leg(1, 10, "ETH", "SPOT"), _leg(2, 10, "ETH", "PERP")]


def _series(inputs: PnlInputs, hours: int) -> pd.DataFrame:
    frame, dropped = strategy_pnl.strategy_equity(
        inputs, since_ms=T0, until_ms=T0 + hours * HOUR_MS)
    assert dropped == []
    return frame.set_index("ts_ms")


# --- the series ---------------------------------------------------------------------------------


def test_the_series_starts_at_the_budget_and_a_flat_hedged_book_stays_there() -> None:
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(),
                     marks=_marks("ETH", [100, 100, 100, 100, 100, 100]))

    frame = _series(inputs, 3)

    assert frame["strategy_equity"].iloc[0] == 90.0
    assert frame["strategy_equity"].tolist() == [90.0, 90.0, 90.0, 90.0]


def test_the_same_price_move_on_the_hedged_legs_cancels() -> None:
    # The spot long gains what the perp short loses: a hedge.
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(),
                     marks=_marks("ETH", [100, 100, 100, 110, 120, 130]))

    frame = _series(inputs, 3)

    assert frame["strategy_equity"].tolist() == pytest.approx([90.0] * 4)


def test_funding_received_on_the_perp_leg_is_added_once_it_has_been_paid() -> None:
    funding = [(2, T0 + h * HOUR_MS + 111, 0.25) for h in range(1, 5)]  # just after the hour
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(), funding=funding,
                     marks=_marks("ETH", [100] * 8))

    frame = _series(inputs, 4)

    assert frame["funding"].tolist() == pytest.approx([0.0, 0.0, 0.25, 0.5, 0.75])
    assert frame["strategy_equity"].tolist() == pytest.approx([90.0, 90.0, 90.25, 90.5, 90.75])


def test_funding_before_the_period_is_not_part_of_it() -> None:
    funding = [(2, T0 - 5 * HOUR_MS, 3.0), (2, T0 + 2 * HOUR_MS + 5, 0.1)]
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(), funding=funding,
                     marks=_marks("ETH", [100] * 6))

    frame = _series(inputs, 3)

    assert frame["funding"].iloc[-1] == pytest.approx(0.1)


def test_fees_paid_inside_the_period_are_subtracted_and_earlier_ones_are_not() -> None:
    fills = [(1, 1, T0 - HOUR_MS, "long", 1.0, 100.0, 0.05),    # before the period: sunk
             (2, 2, T0 - HOUR_MS, "short", 1.0, 100.0, 0.04),
             (3, 3, T0 + 2 * HOUR_MS, "long", 1.0, 100.0, 0.02),    # opened inside the period
             (4, 4, T0 + 2 * HOUR_MS, "short", 1.0, 100.0, 0.01)]
    legs = _pair_legs() + [_leg(3, 11, "SOL", "SPOT"), _leg(4, 11, "SOL", "PERP")]
    legs[2]["opened_at"] = legs[3]["opened_at"] = T0 + 2 * HOUR_MS
    marks = _marks("ETH", [100] * 6) + _marks("SOL", [100] * 6)
    inputs = _inputs(legs=legs, fills=fills, marks=marks)

    frame = _series(inputs, 3)

    assert frame["fees"].tolist() == pytest.approx([0.0, 0.0, 0.03, 0.03])
    assert frame["strategy_equity"].iloc[-1] == pytest.approx(90.0 - 0.03)


def test_an_open_leg_is_marked_unrealised_and_a_closed_leg_keeps_its_realised_result() -> None:
    # The spot leg is sold at 130 at hour 2; the perp short stays open. The prices seen at the
    # boundaries 0..4 are 100, 100, 100, 120, 140 (rows are listed from boundary -2).
    fills = _pair_fills() + [(3, 1, T0 + 2 * HOUR_MS, "short", 1.0, 130.0, 0.0)]
    inputs = _inputs(legs=_pair_legs(), fills=fills,
                     marks=_marks("ETH", [100, 100, 100, 100, 100, 120, 140]))

    frame = _series(inputs, 4)

    # hour 2: spot realised +30, perp still at 100. hour 3 (price 120): +30 - 20 = +10.
    # hour 4 (price 140): +30 - 40 = -10. The sold spot leg no longer follows the price.
    assert frame["trading_pnl"].tolist() == pytest.approx([0.0, 0.0, 30.0, 10.0, -10.0])


def test_a_closed_leg_that_leaves_dust_keeps_moving_with_the_price() -> None:
    # Bought 1.0 and sold 0.99 before the period (the sell was floored to the lot): 0.01 stays.
    dust = _leg(3, 9, "ETH", "SPOT", state="CLOSED")
    dust["closed_at"] = T0 - 5 * HOUR_MS
    fills = _pair_fills() + [(3, 3, T0 - 10 * HOUR_MS, "long", 1.0, 100.0, 0.0),
                             (4, 3, T0 - 5 * HOUR_MS, "short", 0.99, 100.0, 0.0)]
    inputs = _inputs(legs=_pair_legs() + [dust], fills=fills,
                     marks=_marks("ETH", [100, 100, 100, 100, 110, 130]))

    frame = _series(inputs, 3)

    # The hedged pair cancels; the dust (0.01 ETH) gains 0.01 x (130 - 100) = 0.3 by hour 3.
    assert frame["trading_pnl"].iloc[-1] == pytest.approx(0.3)


def test_the_spot_leg_uses_its_own_price_when_the_host_stored_one() -> None:
    # From the boundary 1 on the host stored a spot price of 99 beside the perp's 100.
    marks = _marks("ETH", [100] * 6, spot=[None, None, None, 99.0, 99.0, 99.0])
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(), marks=marks)

    frame = _series(inputs, 3)

    # The long spot is 1 down at its own price, the short perp flat at the perp's.
    assert frame["trading_pnl"].iloc[-1] == pytest.approx(-1.0)


def test_a_deposit_to_the_account_cannot_move_the_series(tmp_path: Path) -> None:
    """The account's equity is not an input. A deposit shows in `equity_snapshots`, which the
    export does not read, so the series with and without it are the same file."""
    folder = _export_folder(tmp_path / "plain")
    with_deposit = _export_folder(tmp_path / "deposit")
    pd.DataFrame({"strategy_id": 1, "ts_ms": [T0, T0 + HOUR_MS, T0 + 2 * HOUR_MS],
                  "total_equity": [128.0, 228.0, 228.1]}).to_csv(
        with_deposit / "frab_equity.csv", index=False)  # +100 deposited at hour 1

    for target in (folder, with_deposit):
        strategy_pnl.build_series_file(target, filename="series.csv", strategy_id=1,
                                       since_ms=T0, expected_budget=90.0)

    a = pd.read_csv(folder / "series.csv")
    b = pd.read_csv(with_deposit / "series.csv")
    pd.testing.assert_frame_equal(a, b)
    assert a["strategy_equity"].max() < 91  # nowhere near 128 or 228


def test_collateral_rows_and_failed_positions_without_fills_are_ignored() -> None:
    collateral = _leg(5, 10, "USDC", "COLLATERAL")
    collateral["entry_price"] = 1.0
    failed = _leg(6, 12, "BTC", "SPOT", state="FAILED")
    legs = _pair_legs() + [collateral, failed]
    funding = [(5, T0 + HOUR_MS + 1, 9.0), (6, T0 + HOUR_MS + 1, 9.0)]  # nothing real pays here
    # A (nonsensical) fill on the COLLATERAL row must not count either.
    fills = _pair_fills() + [(9, 5, T0 - HOUR_MS, "long", 2.4, 1.0, 5.0)]
    inputs = _inputs(legs=legs, fills=fills, funding=funding, marks=_marks("ETH", [100] * 6))

    frame = _series(inputs, 3)

    assert frame["strategy_equity"].tolist() == pytest.approx([90.0] * 4)
    assert frame["fees"].iloc[-1] == 0.0


def test_an_hour_without_a_price_is_dropped_and_reported_not_filled() -> None:
    prices = _marks("ETH", [100] * 3)  # prices for boundaries -2..0 only, then a hole
    prices += _marks("ETH", [100], first_hour=7)
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(), marks=prices)

    frame, dropped = strategy_pnl.strategy_equity(inputs, since_ms=T0, until_ms=T0 + 6 * HOUR_MS)

    assert dropped  # the hours beyond the price's age are named
    assert set(dropped).isdisjoint(frame["ts_ms"])
    assert T0 in set(frame["ts_ms"])


def test_no_price_at_the_first_hour_refuses_the_series() -> None:
    inputs = _inputs(legs=_pair_legs(), fills=_pair_fills(),
                     marks=_marks("ETH", [100] * 3, first_hour=8))

    with pytest.raises(StrategyPnlError, match="first hour"):
        strategy_pnl.strategy_equity(inputs, since_ms=T0, until_ms=T0 + 3 * HOUR_MS)


def test_per_position_breakdown_is_the_result_since_it_opened() -> None:
    funding = [(2, T0 - 10 * HOUR_MS, 0.5), (2, T0 + 5, 0.1)]
    fills = [(1, 1, T0 - HOUR_MS, "long", 1.0, 100.0, 0.02),
             (2, 2, T0 - HOUR_MS, "short", 1.0, 100.0, 0.01)]
    inputs = _inputs(legs=_pair_legs(), fills=fills, funding=funding,
                     marks=_marks("ETH", [100, 100, 100, 110, 110]))

    [row] = strategy_pnl.position_breakdown(inputs, T0 + 2 * HOUR_MS).to_dict("records")

    assert row["funding"] == pytest.approx(0.6)  # since the position opened, not since the period
    assert row["fees"] == pytest.approx(0.03)
    assert row["spot_pnl"] == pytest.approx(10.0) and row["perp_pnl"] == pytest.approx(-10.0)


# --- the budget -------------------------------------------------------------------------


def test_the_budget_is_read_from_the_strategy_params() -> None:
    assert strategy_pnl.budget_from_params(json.dumps({"budget_cap_usdc": 90})) == 90.0
    for bad in ({}, {"budget_cap_usdc": 0}, {"budget_cap_usdc": "90"}, {"budget_cap_usdc": None}):
        with pytest.raises(StrategyPnlError, match="budget_cap_usdc"):
            strategy_pnl.budget_from_params(json.dumps(bad))


def test_a_budget_that_differs_from_the_written_down_capital_is_refused(tmp_path: Path) -> None:
    folder = _export_folder(tmp_path)

    with pytest.raises(StrategyPnlError, match="strategy_capital_usd"):
        strategy_pnl.build_series_file(folder, filename="series.csv", strategy_id=1,
                                       since_ms=T0, expected_budget=120.0)

    assert not (folder / "series.csv").exists()


def test_the_series_file_is_what_the_decay_detector_reads_and_its_percent_is_on_the_budget(
        tmp_path: Path) -> None:
    folder = _export_folder(tmp_path, funding_per_hour=0.09)  # +0.09 per hour on a $90 budget
    strategy_pnl.build_series_file(folder, filename="series.csv", strategy_id=1,
                                   since_ms=T0, expected_budget=90.0)

    curve = paper_check.equity_curve(folder / "series.csv", column="strategy_equity",
                                     strategy_id=1)
    done = paper_check.window_return(curve, datetime(2026, 10, 10, tzinfo=UTC))

    assert curve.iloc[0] == 90.0
    assert done is not None and done.total == pytest.approx(0.09 * 24 / 90)  # 24 hours, on $90


def _export_folder(folder: Path, *, funding_per_hour: float = 0.01) -> Path:
    """The five export files of a hedged pair with funding paid each hour for 24 hours."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_pair_legs()).to_csv(folder / strategy_pnl.LEGS_FILE, index=False)
    pd.DataFrame(_pair_fills(), columns=["fill_id", "position_id", "ts_ms", "side", "qty",
                                         "price", "fee"]).to_csv(
        folder / strategy_pnl.FILLS_FILE, index=False)
    pd.DataFrame([(2, T0 + h * HOUR_MS + 50, funding_per_hour) for h in range(0, 24)],
                 columns=["position_id", "ts_ms", "amount"]).to_csv(
        folder / strategy_pnl.FUNDING_FILE, index=False)
    pd.DataFrame(_marks("ETH", [100] * 30)).to_csv(folder / strategy_pnl.MARKS_FILE, index=False)
    pd.DataFrame({"strategy_id": [1], "params_json": [json.dumps({"budget_cap_usdc": 90})]}
                 ).to_csv(folder / strategy_pnl.BUDGET_FILE, index=False)
    return folder


# --- the export -------------------------------------------------------------------------


def test_the_export_is_select_only_and_safe_inside_a_double_quoted_remote_command() -> None:
    queries = strategy_pnl.export_queries(1, T0)

    assert len(queries) == 5
    for _, sql in queries:
        assert sql.lstrip().lower().startswith("select")
        for word in ("insert", "update", "delete", "drop", "alter", "pragma", "attach"):
            assert word not in sql.lower().split()
        assert '"' not in sql and "$" not in sql and ";" not in sql
    legs, fills, funding, marks, budget = (sql for _, sql in queries)
    assert "instrument<>'COLLATERAL'" in legs and "strategy_id=1" in legs
    assert f"p.closed_at>={T0}" in legs
    assert "abs((select coalesce(sum(case when fl.side='long'" in legs  # dust of closed legs
    assert "from fills" in fills and "from funding_accruals" in funding
    assert "from prices" in marks and "row_number()" in marks  # one freshest row per clock hour
    assert "from strategies where id=1" in budget


# --- the books: strategy money is written down for every book ----------------------------------


def _shipped() -> dict:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "night" / "paper_books.yaml").read_text(encoding="utf-8"))


def test_every_shipped_book_states_the_money_available_to_the_strategy() -> None:
    raw = _shipped()
    books = {**raw["books"], **raw["live_books"]}

    assert books  # the loop below must not be vacuous
    for name, cfg in books.items():
        capital = cfg.get("strategy_capital_usd")
        assert isinstance(capital, int | float) and capital > 0, name
    assert {k: v["strategy_capital_usd"] for k, v in raw["books"].items()} == {
        "strategy-b-v2": 257, "bv2-research-headroom": 257, "trend-live": 3000,
        "bv2-research-headroom-retune": 257, "trend-retune-weekly": 3000}
    # FRAB: the strategy works on 90, whatever else the account holds.
    assert raw["live_books"]["frab"]["strategy_capital_usd"] == 90


def test_a_book_without_strategy_capital_is_refused_at_load(tmp_path: Path) -> None:
    path = tmp_path / "books.yaml"
    path.write_text("books:\n  p:\n    book: trend\n    paper_file: t.csv\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strategy_capital_usd"):
        paper_check.load_books(path)
    with pytest.raises(ValueError, match="strategy_capital_usd"):
        decay.load_all_books(path)


def test_a_live_book_on_account_equity_is_refused_at_load(tmp_path: Path) -> None:
    path = tmp_path / "books.yaml"
    path.write_text(
        "books: {}\nlive_books:\n  frab:\n    book: two_phase\n    paper_file: f.csv\n"
        "    equity_column: total_equity\n    strategy_capital_usd: 90\n", encoding="utf-8")

    with pytest.raises(ValueError, match="equity_from"):
        decay.load_all_books(path)


LATEST_EXPORT = sorted((Path(__file__).resolve().parents[2] / "data" / "paper").glob("2026-*"))


@pytest.mark.parametrize("name", ["strategy-b-v2", "bv2-research-headroom", "trend-live",
                                  "bv2-research-headroom-retune", "trend-retune-weekly"])
def test_a_paper_books_capital_is_what_its_export_shows_at_its_start(name: str) -> None:
    """The paper engine has no money but the strategy's: its equity at the book's FIRST row is
    its capital. (b2_retune replayed its history at launch, so by its `realised_from` the
    equity has moved on: that is the stand's own arithmetic, not a different capital.)"""
    books = decay.load_all_books(_shipped_path())
    cfg = books[name]
    root = Path(__file__).resolve().parents[2] / "data" / "paper"
    csv = next((c for d in sorted(root.glob("2026-*"), reverse=True)
                if (c := d / cfg["paper_file"]).is_file()
                and strategy_id_in(c, cfg["strategy_id"])), None)
    if csv is None:
        pytest.skip(f"no local export of {cfg['paper_file']} holds strategy {cfg['strategy_id']}")
    curve = paper_check.equity_curve(csv, strategy_id=cfg["strategy_id"],
                                     sum_coins=bool(cfg.get("sum_coins")))

    # The first hour's trading and fees move it a little; a wrong capital moves it a lot.
    assert decay.capital_mismatch(cfg, float(curve.iloc[0]), tolerance=0.01) is None


def _shipped_path() -> Path:
    return Path(__file__).resolve().parents[2] / "night" / "paper_books.yaml"


def strategy_id_in(csv: Path, strategy_id: int) -> bool:
    frame = pd.read_csv(csv, usecols=["strategy_id"])
    return bool((frame["strategy_id"] == strategy_id).any())


def test_capital_mismatch_names_both_numbers() -> None:
    cfg = {"strategy_capital_usd": 90}

    assert decay.capital_mismatch(cfg, 90.0, tolerance=1e-6) is None
    assert decay.capital_mismatch(cfg, 90.4, tolerance=0.01) is None
    text = decay.capital_mismatch(cfg, 128.0, tolerance=1e-6)
    assert text is not None and "90" in text and "128" in text


# --- the decay wiring -------------------------------------------------------------------


def _decay_books(tmp_path: Path) -> dict:
    path = tmp_path / "books.yaml"
    path.write_text(
        "books: {}\nlive_books:\n  frab-x:\n    book: two_phase\n    spec: y.yaml\n"
        "    label: FRAB\n"
        "    paper_file: frab_strategy_equity.csv\n    equity_from: strategy_pnl\n"
        "    equity_column: strategy_equity\n    strategy_capital_usd: 90\n    strategy_id: 1\n"
        "    realised_from: 2026-10-10\n", encoding="utf-8")
    return decay.load_all_books(path)


def _stand(session, spec_path, *, end, cut):
    index = pd.date_range("2026-01-01", periods=180, freq="D", tz="UTC")
    return pd.Series([0.001, -0.0005, 0.002] * 60, index=index)


def test_the_decay_percent_of_a_live_book_is_on_the_strategy_money(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    (root / "2026-10-25").mkdir(parents=True)
    hours = 24 * 12
    # The strategy earned +1.8 on 90 (= +2%); the account (not in this file) is far bigger.
    pd.DataFrame({"strategy_id": 1, "ts_ms": [T0 + h * HOUR_MS for h in range(hours + 1)],
                  "strategy_equity": [90.0 + 1.8 * h / hours for h in range(hours + 1)]}
                 ).to_csv(root / "2026-10-25" / "frab_strategy_equity.csv", index=False)

    results, lines = decay.run_decay(None, date(2026, 10, 25), date(2026, 10, 24),
                                     books=_decay_books(tmp_path),
                                     config={"alert_percentile": None, "preregistered": {}},
                                     root=root, stand_fn=_stand)

    [frab] = results
    assert frab.ret.realised == pytest.approx(1.8 / 90)
    assert frab.live


def test_a_live_series_that_does_not_start_at_the_strategy_money_is_not_compared(
        tmp_path: Path) -> None:
    root = tmp_path / "paper"
    (root / "2026-10-25").mkdir(parents=True)
    pd.DataFrame({"strategy_id": 1, "ts_ms": [T0 + h * HOUR_MS for h in range(24 * 5)],
                  "strategy_equity": [128.0 + 0.01 * h for h in range(24 * 5)]}  # account money
                 ).to_csv(root / "2026-10-25" / "frab_strategy_equity.csv", index=False)

    results, lines = decay.run_decay(None, date(2026, 10, 25), date(2026, 10, 24),
                                     books=_decay_books(tmp_path),
                                     config={"alert_percentile": None, "preregistered": {}},
                                     root=root, stand_fn=_stand)

    assert results == []
    assert any("деньги стратегии по записи 90" in ln and "128" in ln for ln in lines)


def test_the_nights_export_builds_the_series_from_the_rows(tmp_path: Path) -> None:
    folder = _export_folder(tmp_path / "day")
    books = _decay_books(tmp_path)

    lines = decay.build_live_series(folder, books)

    assert lines == []
    assert (folder / "frab_strategy_equity.csv").is_file()
    assert pd.read_csv(folder / "frab_strategy_equity.csv")["strategy_equity"].iloc[0] == 90.0


def test_a_series_that_cannot_be_built_is_a_report_line_not_a_failure(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    [line] = decay.build_live_series(folder, _decay_books(tmp_path))

    assert line.startswith("FRAB: ряд денег стратегии не построен")
