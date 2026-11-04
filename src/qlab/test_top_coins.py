"""The coins a top-K rule can ever pick, on synthetic daily volumes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import top_coins


def _volumes(days: int = 90) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    data = {"AAA": np.full(days, 100.0), "BBB": np.full(days, 80.0), "CCC": np.full(days, 60.0),
            # DDD becomes the biggest coin from day 50
            "DDD": np.r_[np.full(50, 10.0), np.full(days - 50, 500.0)]}
    return pd.DataFrame(data, index=index)


def test_the_first_k_by_trailing_value_change_when_a_coin_grows() -> None:
    inside = top_coins.top_k(_volumes(), k=2, window_days=10)
    assert list(inside.columns[inside.iloc[40]]) == ["AAA", "BBB"]
    assert list(inside.columns[inside.iloc[80]]) == ["AAA", "DDD"], "BBB leaves, DDD joins"


def test_a_day_does_not_see_its_own_volume() -> None:
    inside = top_coins.top_k(_volumes(), k=2, window_days=10)
    # the volume jumps on day 50; its 10-day mean passes BBB's only a few days later, and the
    # rank of day 50 itself reads days 40..49
    assert not inside["DDD"].iloc[50]
    first = inside["DDD"].idxmax()
    assert first > pd.Timestamp("2024-01-01") + pd.Timedelta(days=50)


def test_a_coin_without_a_full_window_does_not_compete() -> None:
    volume = _volumes()
    volume.loc[volume.index[:45], "DDD"] = np.nan
    rank = top_coins.trailing_rank(volume, window_days=10)
    assert rank["DDD"].iloc[:55].isna().all(), "listed on day 45: a full window ends on day 55"


def test_coins_ever_in_top_lists_every_coin_that_was_inside_once() -> None:
    assert top_coins.coins_ever_in_top(_volumes(), k=2, window_days=10) == ["AAA", "BBB", "DDD"]
    assert top_coins.coins_ever_in_top(_volumes(), k=1, window_days=10) == ["AAA", "DDD"]
    assert top_coins.coins_ever_in_top(_volumes(), k=4, window_days=10) == [
        "AAA", "BBB", "CCC", "DDD"]


def test_only_the_coins_with_a_spot_leg_compete_when_among_is_given() -> None:
    ever = top_coins.coins_ever_in_top(_volumes(), k=1, window_days=10, among=["BBB", "CCC"])
    assert ever == ["BBB"], "AAA and DDD have no spot leg: they never take a place"
