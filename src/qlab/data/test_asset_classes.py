"""Tests for `qlab.data.asset_classes`: the rule, the point-in-time table and the panel labels."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qlab.data import asset_classes as ac

RULE_PATH = Path(__file__).resolve().parents[3] / "classification" / "hyperliquid-xyz.rule.yaml"
TABLE_PATH = Path(__file__).resolve().parents[3] / "classification" / "hyperliquid-xyz.yaml"


@pytest.fixture(scope="module")
def rule() -> ac.ClassRule:
    return ac.load_rule(RULE_PATH)


def _note(category: str, description: str) -> dict:
    return {"category": category, "description": description}


# ---- the rule -------------------------------------------------------------------------------


def test_single_stock_gets_asset_class_stock_and_the_country_of_its_exchange(rule) -> None:
    text = "TSLA references 1 share of Tesla, Inc. common stock (Nasdaq: TSLA), quoted in USD."
    got = ac.classify(_note("stocks", text), rule)
    assert (got.asset_class, got.class_key, got.country) == ("stock", "stock:US", "US")


def test_depositary_share_takes_the_country_of_the_exchange_it_is_priced_on(rule) -> None:
    text = (
        "BABA references 1 Alibaba Group Holding Limited ADS (NYSE: BABA), representing 8 "
        "ordinary shares (HKEX: 9988), quoted in USD."
    )
    assert ac.classify(_note("stocks", text), rule).class_key == "stock:US"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("X references 1 Hyundai Motor Company common share (KRX: 005380).", "stock:KR"),
        ("X references 1 Kioxia Holdings Corporation ordinary share (TSE: 285A).", "stock:JP"),
        ("X references 1 MiniMax Group Inc. Class A ordinary share (HKEX: 0100).", "stock:HK"),
        ("X references 1 CXMT Corporation A-share (SSE STAR: 688825).", "stock:CN"),
        ("X references 1 GigaDevice Semiconductor Inc. A-share (SSE: 603986).", "stock:CN"),
    ],
)
def test_each_exchange_maps_to_its_country(rule, text: str, key: str) -> None:
    assert ac.classify(_note("stocks", text), rule).class_key == key


def test_the_first_exchange_named_decides_not_the_longest_prefix_match(rule) -> None:
    text = "X references 1 share (NYSE Arca: EWY), also (Nasdaq: EWY)."
    assert ac.classify(_note("stocks", text), rule).country == "US"
    both = "X references an ADS (Nasdaq: ASML), the Euronext line (TSE: 0000)."
    assert ac.classify(_note("stocks", both), rule).country == "US"


@pytest.mark.parametrize(
    "text",
    [
        "XLE references 1 share of the Energy Select Sector SPDR ETF (NYSE Arca: XLE). The fund",
        "SOXL references the Direxion Daily Semiconductor Bull 3X Shares (NYSE Arca: SOXL). The "
        "fund seeks daily investment results.",
        "BOT references RoboStrategy, Inc. (Nasdaq: BOT), a closed-end investment company.",
    ],
)
def test_funds_are_not_single_stocks(rule, text: str) -> None:
    got = ac.classify(_note("stocks", text), rule)
    assert (got.asset_class, got.class_key) == ("fund", "fund")


def test_a_preferred_share_is_its_own_kind(rule) -> None:
    text = "STRC references 1 share of Strategy Inc's Variable Rate Series A Perpetual Stretch " \
           "Preferred Stock (Nasdaq: STRC), quoted in USD."
    assert ac.classify(_note("stocks", text), rule).asset_class == "preferred"


@pytest.mark.parametrize(
    ("category", "kind"),
    [("commodities", "commodity"), ("fx", "fx"), ("indices", "index"), ("preipo", "preipo")],
)
def test_other_categories_map_straight_to_their_kind(rule, category: str, kind: str) -> None:
    got = ac.classify(_note(category, "some text"), rule)
    assert (got.asset_class, got.class_key, got.country) == (kind, kind, None)


def test_no_annotation_a_placeholder_or_an_unknown_category_is_unclassified(rule) -> None:
    assert ac.classify(None, rule).asset_class == ac.UNCLASSIFIED
    assert ac.classify(_note("indices", "coming soon"), rule).asset_class == ac.UNCLASSIFIED
    unknown = ac.classify(_note("weather", "Rain in Oslo"), rule)
    assert unknown.asset_class == ac.UNCLASSIFIED
    assert "not in the rule" in (unknown.note or "")


def test_a_stock_with_no_known_exchange_is_kept_visible_not_dropped_silently(rule) -> None:
    got = ac.classify(_note("stocks", "X references 1 share of Y Corp."), rule)
    assert (got.asset_class, got.class_key) == ("stock", "stock:unknown")
    assert got.note


def test_classification_matches_by_kind_or_by_key(rule) -> None:
    got = ac.classify(_note("stocks", "X (Nasdaq: X) common stock"), rule)
    assert got.matches(["stock"]) and got.matches(["stock:US"])
    assert not got.matches(["stock:KR"]) and not got.matches(["fund"])


# ---- the table, point in time ---------------------------------------------------------------


def _document(rule, snapshots, facts) -> dict:
    return ac.build_table(rule, snapshots, facts)


_DELISTED = ac.ListingFacts(date(2026, 3, 1), "first_candle", date(2026, 6, 1), "seen_delisted")
_DELISTED_JAN4 = ac.ListingFacts(
    date(2026, 1, 1), "first_candle", date(2026, 1, 4), "seen_delisted"
)
STOCK = _note("stocks", "A references 1 share (Nasdaq: A) common stock")
FUND = _note("stocks", "B references 1 share of the B ETF (NYSE Arca: B). The fund ...")


def test_class_is_none_before_listing_known_after_and_none_once_delisted(rule) -> None:
    facts = {
        "xyz:A": _DELISTED
    }
    table = ac.ClassTable(_document(rule, {date(2026, 7, 1): {"xyz:A": STOCK}}, facts))
    assert table.class_at("xyz:A", date(2026, 2, 28)) is None
    assert table.class_at("xyz:A", date(2026, 3, 1)).class_key == "stock:US"
    assert table.class_at("xyz:A", date(2026, 5, 31)).class_key == "stock:US"
    assert table.class_at("xyz:A", date(2026, 6, 1)) is None
    assert table.class_at("xyz:UNKNOWN", date(2026, 5, 1)) is None


def test_a_delisted_instrument_keeps_its_class_history(rule) -> None:
    facts = {
        "xyz:A": _DELISTED
    }
    table = ac.ClassTable(_document(rule, {date(2026, 7, 1): {"xyz:A": STOCK}}, facts))
    history = table.class_history("xyz:A")
    assert [(d, c.class_key) for d, c in history] == [(date(2026, 3, 1), "stock:US")]


def test_a_changed_annotation_applies_from_the_bar_after_it_was_seen_never_backwards(rule) -> None:
    facts = {"xyz:A": ac.ListingFacts(date(2026, 1, 1), "first_candle")}
    snapshots = {date(2026, 4, 1): {"xyz:A": STOCK}, date(2026, 5, 10): {"xyz:A": FUND}}
    table = ac.ClassTable(_document(rule, snapshots, facts))
    assert table.class_at("xyz:A", date(2026, 5, 10)).class_key == "stock:US"
    assert table.class_at("xyz:A", date(2026, 5, 11)).class_key == "fund"
    assert table.class_at("xyz:A", date(2026, 2, 1)).class_key == "stock:US"


def test_an_unchanged_annotation_adds_no_history_entry(rule) -> None:
    facts = {"xyz:A": ac.ListingFacts(date(2026, 1, 1), "first_candle")}
    snapshots = {date(2026, 4, 1): {"xyz:A": STOCK}, date(2026, 4, 2): {"xyz:A": STOCK}}
    table = ac.ClassTable(_document(rule, snapshots, facts))
    assert len(table.class_history("xyz:A")) == 1


def test_an_instrument_first_described_later_is_unclassified_until_then(rule) -> None:
    facts = {"xyz:A": ac.ListingFacts(date(2026, 1, 1), "first_candle")}
    snapshots = {date(2026, 4, 1): {"xyz:A": None}, date(2026, 5, 1): {"xyz:A": STOCK}}
    table = ac.ClassTable(_document(rule, snapshots, facts))
    assert table.class_at("xyz:A", date(2026, 3, 1)).asset_class == ac.UNCLASSIFIED
    assert table.class_at("xyz:A", date(2026, 5, 2)).class_key == "stock:US"


def test_build_table_is_deterministic_and_round_trips_through_yaml(rule, tmp_path) -> None:
    facts = {"xyz:A": ac.ListingFacts(date(2026, 1, 1), "first_candle")}
    document = _document(rule, {date(2026, 4, 1): {"xyz:A": STOCK}}, facts)
    path = tmp_path / "t.yaml"
    ac.write_table(document, path)
    first = path.read_bytes()
    ac.write_table(_document(rule, {date(2026, 4, 1): {"xyz:A": STOCK}}, facts), path)
    assert path.read_bytes() == first
    assert ac.load_table(path).class_at("xyz:A", date(2026, 4, 2)).class_key == "stock:US"


def test_build_table_without_snapshots_fails_loudly(rule) -> None:
    with pytest.raises(ValueError, match="no annotation snapshots"):
        ac.build_table(rule, {}, {})


# ---- listing facts from the record ----------------------------------------------------------


def _listing(path: Path, day: str, flags: dict[str, bool]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    universe = [{"name": n, **({"isDelisted": True} if d else {})} for n, d in flags.items()]
    (path / f"{day}.json").write_text(json.dumps({"universe": universe}), encoding="utf-8")


def test_listing_facts_use_the_first_candle_else_the_first_snapshot_and_find_delisting(
    tmp_path,
) -> None:
    record = tmp_path / "rec"
    for day, flags in [
        ("2026-10-01", {"xyz:A": False, "xyz:B": False, "xyz:C": True, "xyz:R": True}),
        ("2026-10-02", {"xyz:A": False, "xyz:B": True, "xyz:C": True, "xyz:R": False}),
        ("2026-10-03", {"xyz:A": False, "xyz:B": True, "xyz:C": True, "xyz:R": False,
                        "xyz:N": False}),
    ]:
        _listing(record / "meta", day, flags)
    candles = record / "candles" / "1d"
    candles.mkdir(parents=True)
    index = pd.date_range("2026-09-01", periods=3, freq="1D", tz="UTC", name="timestamp")
    pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index).to_parquet(candles / "xyz:A.parquet")
    facts = ac.listing_facts(record, raw_dir=None)
    assert facts["xyz:A"] == ac.ListingFacts(date(2026, 9, 1), "first_candle")
    assert facts["xyz:B"].delisted_since == date(2026, 10, 2)
    assert facts["xyz:B"].delisted_basis == "seen_delisted"
    assert facts["xyz:C"].delisted_since == date(2026, 10, 1)
    assert facts["xyz:C"].delisted_basis == "on_or_before_first_snapshot"
    assert facts["xyz:R"].delisted_since is None  # delisted, then listed again
    assert facts["xyz:N"].listed_from == date(2026, 10, 3)
    assert facts["xyz:N"].basis == "first_listing_snapshot"


def test_annotation_snapshots_round_trip(tmp_path) -> None:
    ac.save_annotation_snapshot(tmp_path, date(2026, 10, 25), {"xyz:A": STOCK, "xyz:B": None})
    assert ac.load_annotation_snapshots(tmp_path) == {
        date(2026, 10, 25): {"xyz:A": STOCK, "xyz:B": None}
    }


# ---- labels over a panel --------------------------------------------------------------------


def _small_table(rule) -> ac.ClassTable:
    facts = {
        "xyz:A": ac.ListingFacts(date(2026, 1, 3), "first_candle"),
        "xyz:F": ac.ListingFacts(date(2026, 1, 1), "first_candle"),
        "xyz:D": _DELISTED_JAN4,
        "xyz:U": ac.ListingFacts(date(2026, 1, 1), "first_candle"),
    }
    snapshots = {
        date(2026, 1, 6): {
            "xyz:A": STOCK, "xyz:F": FUND, "xyz:D": STOCK, "xyz:U": None,
        }
    }
    return ac.ClassTable(_document(rule, snapshots, facts))


def test_labels_frame_is_point_in_time_per_bar(rule) -> None:
    table = _small_table(rule)
    index = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    labels = table.labels_frame(index, ["xyz:A", "xyz:F", "xyz:D", "xyz:U", "xyz:ZZ"])
    assert labels["xyz:A"].tolist() == [None, None, "stock:US", "stock:US", "stock:US"]
    assert labels["xyz:F"].tolist() == ["fund"] * 5
    assert labels["xyz:D"].tolist() == ["stock:US", "stock:US", "stock:US", None, None]
    assert labels["xyz:U"].isna().all()  # unclassified belongs to no class
    assert labels["xyz:ZZ"].isna().all()  # not in the table


def test_member_mask_accepts_kinds_and_keys_and_excludes_everything_else(rule) -> None:
    table = _small_table(rule)
    index = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    cols = ["xyz:A", "xyz:F", "xyz:D", "xyz:U"]
    stocks = table.member_mask(index, cols, ["stock"])
    assert stocks["xyz:A"].tolist() == [False, False, True, True, True]
    assert not stocks["xyz:F"].any() and not stocks["xyz:U"].any()
    by_key = table.member_mask(index, cols, ["stock:KR"])
    assert not by_key.any().any()
    both = table.member_mask(index, cols, ["fund", "stock:US"])
    assert both["xyz:F"].all()


# ---- the committed rule and table -----------------------------------------------------------


def test_the_committed_rule_loads_and_names_its_sources() -> None:
    loaded = ac.load_rule(RULE_PATH)
    assert loaded.version == "2026-10-25.1"
    assert "annotation" in loaded.sources


def test_the_committed_table_agrees_with_the_rule() -> None:
    document = yaml.safe_load(TABLE_PATH.read_text(encoding="utf-8"))
    loaded = ac.load_rule(RULE_PATH)
    assert document["rule_version"] == loaded.version
    table = ac.load_table(TABLE_PATH)
    # The deployer's own words for single instruments, as the table stored them.
    assert table.class_at("xyz:AAPL", date(2026, 6, 1)).class_key == "stock:US"
    assert table.class_at("xyz:EWJ", date(2026, 6, 1)).class_key == "fund"
    assert table.class_at("xyz:GOLD", date(2026, 6, 1)).class_key == "commodity"
    # Re-applying the rule to the stored text gives the stored class.
    for name, entry in document["instruments"].items():
        last = entry["history"][-1]
        if last.get("underlying") and last["asset_class"] != ac.UNCLASSIFIED:
            again = ac.classify(
                {"category": last["venue_category"], "description": last["underlying"]}, loaded
            )
            assert again.class_key == last["class_key"], name
