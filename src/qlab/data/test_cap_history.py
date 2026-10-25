"""The cap-history export must rank exactly as the stand's market-cap rule did (T52b)."""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qlab.data.cap_history import (
    build_cap_history,
    hyperliquid_perp_names,
    ranked_snapshot,
)
from qlab.data.sources import coinmarketcap
from qlab.strategies.live.trend import _top_k_by_market_cap_mask


def _row(rank: int, symbol: str, cap: float | None, tags: tuple[str, ...] = ()) -> dict:
    return {"cmcRank": rank, "symbol": symbol, "tags": list(tags), "market_cap_usd": cap}


def _write_store(store: Path, snapshots: dict[str, list[dict]]) -> None:
    store.mkdir(parents=True, exist_ok=True)
    for day, rows in snapshots.items():
        payload = json.dumps({"date": day, "rows": rows})
        (store / f"{day}.json").write_text(payload, encoding="utf-8")


def _frame(rows: list[dict], snapshot: str = "2025-01-05") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["snapshot"] = pd.Timestamp(snapshot, tz="UTC")
    return frame


NAMES = {"BTC", "ETH", "SOL", "kPEPE", "DOGE", "ADA"}


def test_symbols_map_to_perps_with_unit_prefixes_and_unlisted_coins_are_dropped():
    rows = _frame([_row(1, "BTC", 900.0), _row(2, "NOPERP", 800.0), _row(3, "PEPE", 700.0)])
    assert ranked_snapshot(rows, NAMES) == [["BTC", 900.0, 1], ["kPEPE", 700.0, 3]]


def test_stablecoins_are_dropped_by_tag_and_the_rest_keep_their_order():
    rows = _frame([
        _row(1, "BTC", 900.0), _row(2, "USDT", 850.0, ("stablecoin",)),
        _row(3, "ETH", 800.0), _row(4, "SOL", 700.0),
    ])
    assert [r[0] for r in ranked_snapshot(rows, NAMES | {"USDT"})] == ["BTC", "ETH", "SOL"]


def test_a_symbol_resolves_to_its_highest_ranked_coin_before_the_exclusion():
    """A stablecoin outranking the real coin of its symbol removes the symbol (as in the stand)."""
    rows = _frame([
        _row(1, "BTC", 900.0), _row(2, "DAI", 850.0, ("stablecoin",)), _row(3, "DAI", 100.0),
        _row(4, "ETH", 800.0),
    ])
    assert [r[0] for r in ranked_snapshot(rows, NAMES | {"DAI"})] == ["BTC", "ETH"]


def test_ranking_follows_market_cap_not_the_listed_rank_and_ties_go_by_name():
    rows = _frame([_row(1, "ETH", 500.0), _row(2, "BTC", 900.0), _row(3, "SOL", 500.0),
                   _row(4, "ADA", None)])
    assert ranked_snapshot(rows, NAMES) == [["BTC", 900.0, 2], ["ETH", 500.0, 1], ["SOL", 500.0, 3]]


def test_only_the_top_n_are_kept():
    rows = _frame([_row(i + 1, s, 1000.0 - i) for i, s in enumerate(["BTC", "ETH", "SOL", "ADA"])])
    assert [r[0] for r in ranked_snapshot(rows, NAMES, top=2)] == ["BTC", "ETH"]


def test_the_file_starts_at_the_last_snapshot_strictly_before_the_window():
    days = ("2024-12-22", "2024-12-29", "2025-01-05", "2025-01-12")
    history = pd.concat([_frame([_row(1, "BTC", 1.0)], d) for d in days], ignore_index=True)
    doc = build_cap_history(history, NAMES, date(2025, 1, 1))
    assert [s["date"] for s in doc["snapshots"]] == ["2024-12-29", "2025-01-05", "2025-01-12"]
    # a snapshot dated ON the window start is not "before" it: the first bar cannot see it
    on_start = build_cap_history(history, NAMES, date(2025, 1, 5))
    assert [s["date"] for s in on_start["snapshots"]] == ["2024-12-29", "2025-01-05", "2025-01-12"]
    assert doc["start"] == "2025-01-01" and doc["top"] == 80


def test_an_empty_store_fails_loudly():
    with pytest.raises(ValueError, match="no CoinMarketCap snapshots"):
        build_cap_history(pd.DataFrame(), NAMES, date(2025, 1, 1))


def test_perp_names_come_from_the_cached_history_files_and_spot_columns_are_left_out(tmp_path):
    for name in ("BTC__2025-01-01__2026-10-24.json", "BTC__2025-01-01__2026-10-24.parquet",
                 "kPEPE__2025-01-01__2026-10-24.json", "AVAX-SPOT__2025-01-01__2026-09-20.json"):
        (tmp_path / name).write_text("{}")
    assert hyperliquid_perp_names(tmp_path) == {"BTC", "kPEPE"}


def test_the_ranking_of_the_latest_visible_snapshot_equals_the_stand_mask_on_every_bar(tmp_path):
    """The exported ranking, read with the stand's visibility rule (a snapshot is visible from the
    first bar strictly after its date), selects the same coins as `_top_k_by_market_cap_mask`."""
    columns = ["ADA", "BTC", "DOGE", "ETH", "SOL", "kPEPE"]
    snapshots = {
        "2025-01-05": [
            _row(1, "BTC", 900.0), _row(2, "USDT", 850.0, ("stablecoin",)), _row(3, "ETH", 800.0),
            _row(4, "PEPE", 700.0), _row(5, "NOPERP", 650.0), _row(6, "SOL", 600.0),
            _row(7, "DOGE", 500.0), _row(8, "ADA", 400.0),
        ],
        "2025-01-12": [
            _row(1, "BTC", 900.0), _row(2, "SOL", 880.0), _row(3, "ADA", 870.0),
            _row(4, "ETH", 860.0, ("stablecoin",)), _row(5, "ETH", 100.0), _row(6, "DOGE", 500.0),
            _row(7, "PEPE", 450.0), _row(8, "NOPERP", 440.0),
        ],
    }
    store = tmp_path / "cmc"
    _write_store(store, snapshots)
    history = coinmarketcap.load_history(store)
    doc = build_cap_history(history, set(columns), date(2025, 1, 6))
    index = pd.date_range("2025-01-04", "2025-01-16", freq="D", tz="UTC")
    prices = pd.DataFrame(1.0, index=index, columns=columns)
    for k in (2, 3, 4):
        mask = _top_k_by_market_cap_mask(prices, k, exclude_tags=("stablecoin",), store=store)
        for bar in index:
            visible = [s for s in doc["snapshots"] if pd.Timestamp(s["date"], tz="UTC") < bar]
            expected = {r[0] for r in visible[-1]["ranked"][:k]} if visible else set()
            got = {c for c in columns if bool(mask.loc[bar, c])}
            assert got == expected, (k, bar)


def test_the_export_script_writes_a_deterministic_document(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_cap_history", Path(__file__).parents[3] / "scripts" / "export_cap_history.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    store, raw, out = tmp_path / "cmc", tmp_path / "raw", tmp_path / "out" / "cap.json"
    _write_store(store, {"2024-12-29": [_row(1, "BTC", 5.0)], "2025-01-05": [_row(1, "ETH", 4.0)]})
    raw.mkdir()
    (raw / "BTC__2025-01-01__2026-10-24.json").write_text("{}")
    (raw / "ETH__2025-01-01__2026-10-24.json").write_text("{}")
    args = ["--store", str(store), "--hl-raw", str(raw), "--out", str(out)]
    assert script.main(args) == 0
    first = out.read_bytes()
    assert script.main(args) == 0 and out.read_bytes() == first
    doc = json.loads(first)
    assert [s["date"] for s in doc["snapshots"]] == ["2024-12-29", "2025-01-05"]
    assert doc["snapshots"][1]["ranked"] == [["ETH", 4.0, 1]]
