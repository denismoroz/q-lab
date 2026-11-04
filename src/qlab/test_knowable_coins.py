"""The knowable-coins check (`qlab.knowable_coins`, docs/ONE_CHECK.md): a hand-picked coin list is
judged from the start of the data when every coin of it was an obvious pick on every past day.

The volumes below are made by hand: a coin's daily traded value is a number per day, so which
coin was in the top on which day is arithmetic a reader can follow."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qlab import knowable_coins as kc
from qlab import twin
from qlab.pipeline.spec import StrategySpec, load_spec

LISTING = [c for coin in ("BTC", "ETH", "SOL", "AVAX", "DOGE") for c in (coin, f"{coin}-SPOT")]
ROOT = Path(__file__).resolve().parents[2]
CONFIG = kc.KnowableConfig(top_k=2, window_days=5, raw_dir=Path("data/raw"), interval="1d",
                           source="night/knowable_coins.yaml")
FIRST = date(2020, 1, 1)
DAYS = 60


def _frame(coins: dict[str, dict[int, float]], days: int = DAYS, first: date = FIRST
           ) -> pd.DataFrame:
    """Daily traded value, one column per coin. `coins[c]` maps a day number to the value from
    that day on (the last key at or before the day wins); a coin has NO value before its first key
    (it is not listed yet)."""
    index = pd.date_range(pd.Timestamp(first), periods=days, freq="D")
    columns = {}
    for coin, steps in coins.items():
        values = []
        for day in range(days):
            keys = [k for k in steps if k <= day]
            values.append(float("nan") if not keys else steps[max(keys)])
        columns[coin] = values
    return pd.DataFrame(columns, index=index)


def _day(n: int) -> date:
    return (pd.Timestamp(FIRST) + pd.Timedelta(days=n)).date()


def _market(**extra: dict[int, float]) -> pd.DataFrame:
    """BTC and ETH always the two biggest; two filler coins below them; plus `extra`."""
    return _frame({"BTC": {0: 1000.0}, "ETH": {0: 900.0}, "F1": {0: 10.0}, "F2": {0: 9.0},
                   **extra})


def _check(frame: pd.DataFrame, coins: list[str]):
    return kc.check(frame, coins, venue="binance", start=_day(0), end=_day(DAYS - 1),
                    config=CONFIG)


def test_btc_and_eth_pass_on_every_day_the_data_can_rank() -> None:
    result = _check(_market(), ["BTC", "ETH"])
    assert result.from_start and result.known_from == result.checked_from == _day(5)
    assert result.coins == {"BTC": _day(5), "ETH": _day(5)}
    assert "BTC, ETH" in result.plain and "история честная" in result.plain


def test_the_spot_column_is_its_coin() -> None:
    result = _check(_market(), ["BTC-SPOT", "BTC", "ETH-SPOT", "ETH"])
    assert list(result.coins) == ["BTC", "ETH"] and result.from_start


def test_a_coin_that_joins_the_top_late_is_judged_from_the_day_it_joined() -> None:
    """NEW is listed from day 0 with the value of a filler and becomes the biggest on day 30."""
    frame = _market(NEW={0: 5.0, 30: 5000.0})
    result = _check(frame, ["BTC", "NEW"])
    assert not result.from_start
    # the trailing mean of days 25..29 ends on day 29; NEW's first big day is 30, so the mean of
    # days 26..30 is the first to reach the top two: day 31 sees it (one big day of five is enough:
    # 1000+... > ETH's 900 only when the mean passes it)
    assert result.coins["NEW"] is not None and result.coins["NEW"] > _day(30)
    assert result.known_from == result.coins["NEW"]
    assert "NEW" in result.plain and "вошла" in result.plain


def test_a_coin_listed_after_the_window_start_is_known_only_after_its_first_full_window() -> None:
    frame = _market(LATE={20: 5000.0})
    result = _check(frame, ["BTC", "LATE"])
    assert result.listed["LATE"] == _day(20)
    assert result.coins["LATE"] == _day(25)  # five full days of candles: days 20..24, seen on 25
    assert result.known_from == _day(25) and not result.from_start
    assert "появилась" in result.plain and str(_day(20)) in result.plain


def test_a_day_never_sees_its_own_volume() -> None:
    """A coin whose first big day is day 30 cannot be known on day 30: the window ends on 29."""
    frame = _market(SPIKE={0: 1.0, 30: 1e9})
    result = _check(frame, ["SPIKE"])
    assert result.coins["SPIKE"] != _day(30)
    assert result.coins["SPIKE"] == _day(31)  # one huge day in the mean of five beats BTC


def test_a_coin_that_never_reaches_the_top_leaves_the_list_dated_by_hand() -> None:
    result = _check(_market(), ["BTC", "F2"])
    assert result.known_from is None and result.coins["F2"] is None and not result.from_start
    assert "F2" in result.plain and "вручную" in result.plain
    assert kc.list_day(result, date(2026, 10, 3)) == date(2026, 10, 3)


def test_a_coin_that_left_the_top_later_is_still_a_knowable_pick() -> None:
    """Survivorship is choosing coins that rose into view; one that faded is no look-ahead."""
    frame = _market(OLD={0: 5000.0, 40: 1.0})
    result = _check(frame, ["BTC", "OLD"])
    assert result.coins["OLD"] == _day(5) and result.known_from == _day(5)


def test_a_coin_the_venue_does_not_list_is_not_knowable() -> None:
    result = _check(_market(), ["BTC", "NOPE"])
    assert result.coins["NOPE"] is None and result.known_from is None


def test_a_gap_in_a_coins_window_leaves_it_unranked_that_day_not_zero() -> None:
    frame = _market()
    frame.loc[frame.index[10], "ETH"] = float("nan")
    result = _check(frame, ["ETH"])
    assert result.coins["ETH"] == _day(5)  # entered before the gap; the gap changes nothing here
    frame2 = _market()
    frame2.loc[frame2.index[2], "ETH"] = float("nan")
    assert _check(frame2, ["ETH"]).coins["ETH"] == _day(8)  # the first full window is days 3..7


def test_list_day_takes_the_earlier_of_the_knowable_day_and_the_hand_picked_one() -> None:
    late = _check(_market(LATE={20: 5000.0}), ["BTC", "LATE"])
    assert kc.list_day(late, None) == _day(25)
    assert kc.list_day(late, date(2020, 1, 10)) == date(2020, 1, 10)
    assert kc.list_day(late, date(2030, 1, 1)) == _day(25)
    start = _check(_market(), ["BTC", "ETH"])
    assert kc.list_day(start, date(2026, 10, 3)) is None
    assert kc.list_day(None, date(2026, 10, 3)) == date(2026, 10, 3)


def test_the_config_names_its_numbers_and_their_reasons() -> None:
    config = kc.load_config(root=ROOT)
    assert (config.top_k, config.window_days) == (10, 30)
    text = (ROOT / kc.CONFIG).read_text(encoding="utf-8")
    assert "WINDOW_DAYS" in text and "top25" in text  # each number carries its source


def test_the_newest_longest_file_of_a_coin_is_read(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    for name, price in (("BTC__2024-01-01__2024-01-02", 1.0), ("BTC__2024-01-01__2024-01-03", 2.0),
                        ("BTC-SPOT__2024-01-01__2024-01-03", 9.0)):
        pd.DataFrame({"price": price, "volume": 10.0}, index=index).to_parquet(
            tmp_path / f"{name}.parquet")
    value = kc.read_daily_value(tmp_path)
    assert list(value.columns) == ["BTC"] and value["BTC"].iloc[0] == 20.0


# --------------------------------------------------------------------------
# In the twin (the idea's one check)


def _bv2_pool() -> StrategySpec:
    return load_spec(ROOT / "specs" / "research" / "bv2" / "bv2-research-shared-pool-hl.yaml")


def _volume_for(spec: StrategySpec, late: dict[str, int] | None = None) -> pd.DataFrame:
    """Years of daily value for BTC, ETH, SOL, AVAX and two fillers on the twin's window; a coin in
    `late` is listed that many days after the archive start."""
    start = pd.Timestamp(twin.BINANCE_ARCHIVE_START)
    days = (pd.Timestamp(spec.data.end) - start).days + 1
    index = pd.date_range(start, periods=days, freq="D")
    base = {"BTC": 1000.0, "ETH": 900.0, "SOL": 800.0, "AVAX": 700.0, "F1": 1.0, "F2": 1.0}
    frame = pd.DataFrame({c: [v] * days for c, v in base.items()}, index=index)
    for coin, offset in (late or {}).items():
        frame.loc[frame.index < start + pd.Timedelta(days=offset), coin] = float("nan")
    return frame


def _twin(spec: StrategySpec, volume: pd.DataFrame) -> StrategySpec:
    config = kc.KnowableConfig(top_k=4, window_days=30, raw_dir=Path("data/raw"), interval="1d",
                               source="night/knowable_coins.yaml")
    result = twin.causal_twin(
        spec, base=load_spec(ROOT / "specs" / "bv2-live.yaml"),
        spec_path=ROOT / "specs" / "research" / "bv2" / "bv2-research-shared-pool-hl.yaml",
        base_path=ROOT / "specs" / "bv2-live.yaml",
        venue_names=LISTING,
        volume_of=lambda venue: volume, knowable_config=config)
    assert result.reason is None, result.reason
    return result.spec


def test_a_list_knowable_on_every_day_is_judged_from_the_start_of_the_data() -> None:
    spec = _bv2_pool()
    made = _twin(spec, _volume_for(spec))
    assert made.instruments_chosen_at is None and made.judged_from is None
    known = made.coins_knowable
    assert known.from_start and known.known_from == date(2019, 10, 8)
    assert set(known.coins) == {"BTC", "ETH", "SOL", "AVAX"}
    assert "coins_knowable" in made.sources and "instruments_chosen_at" not in made.sources
    assert "judged from the start of the data" in made.params_fixed_evidence
    assert made.params_fixed_at > date(2019, 10, 8)  # the code's first choice is the warm-up


def test_a_list_with_a_late_coin_is_judged_from_the_day_it_became_knowable() -> None:
    spec = _bv2_pool()
    made = _twin(spec, _volume_for(spec, late={"AVAX": 400}))
    day = date(2019, 9, 8) + pd.Timedelta(days=430).to_pytimedelta()
    assert made.coins_knowable.known_from == day and not made.coins_knowable.from_start
    assert made.instruments_chosen_at == made.judged_from == day
    assert made.sources["instruments_chosen_at"].startswith("night/knowable_coins.yaml")


def test_a_list_that_was_never_knowable_keeps_its_hand_picked_date() -> None:
    spec = _bv2_pool()
    volume = _volume_for(spec)
    volume["AVAX"] = 0.5  # never among the top four
    made = _twin(spec, volume)
    assert made.coins_knowable.known_from is None
    assert made.instruments_chosen_at == made.judged_from == date(2025, 6, 1)


def test_without_the_venues_data_the_list_is_dated_by_hand_as_before() -> None:
    spec = _bv2_pool()
    result = twin.causal_twin(
        spec, base=load_spec(ROOT / "specs" / "bv2-live.yaml"),
        venue_names=LISTING, volume_of=lambda venue: None)
    assert result.spec.coins_knowable is None
    assert result.spec.judged_from == date(2025, 6, 1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_config_with_no_top_is_refused(tmp_path: Path, top_k: int) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(f"top_k: {top_k}\nwindow_days: 30\nraw_dir: data/raw\n", encoding="utf-8")
    with pytest.raises(ValueError):
        kc.load_config(path)


def test_the_idea_page_says_it_in_one_plain_line(tmp_path: Path) -> None:
    from qlab.plain import venue
    from qlab.plain.specs import SpecView

    known = _check(_market(), ["BTC", "ETH"])
    kc.remember("some-idea", known, store=tmp_path)
    assert kc.recall("some-idea", store=tmp_path) == known and kc.recall("other", tmp_path) is None
    view = SpecView(idea_id="some-idea", title="t", code_ref="x:Y", params={},
                    data={"source": "binance", "start": "2019-09-08", "end": "2026-10-30"},
                    costs={})
    lines = venue.describe(view, coins_line=known.plain)["lines"]
    assert known.plain in lines
    from_file = SpecView(idea_id="i", title="t", code_ref="x:Y", params={}, data={}, costs={},
                         extras={"coins_knowable": {"plain": "из файла"}})
    assert "из файла" in venue.describe(from_file)["lines"]
