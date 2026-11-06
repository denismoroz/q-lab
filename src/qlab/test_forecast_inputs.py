"""Forecast pilot 2: the nulls move labels as declared; the inputs read only what had ended."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import forecast_inputs as fi

HOUR = fi.HOUR
DAY = fi.DAY


def _hours(days: int) -> np.ndarray:
    return pd.date_range("2022-01-01", periods=days * 24, freq="h", tz="UTC").as_unit(
        "ns").asi8


def test_other_day_null_takes_labels_from_a_far_day_at_another_hour() -> None:
    tau = _hours(200)
    rng = np.random.default_rng(0)
    src = fi.other_day_source(tau, rng)
    gap = np.abs(tau[src] // DAY - tau // DAY)
    assert (gap >= 7).all()
    # inside a day the shift is one constant 1..23 hours, never zero
    shift = ((tau[src] // HOUR) - (tau // HOUR)) % 24
    per_day = pd.Series(shift).groupby(tau // DAY).nunique()
    assert (per_day == 1).all()
    assert (shift != 0).all()
    # every label is used exactly once on complete days
    assert sorted(src) == list(range(len(tau)))


def test_within_day_null_stays_in_the_day_and_moves_the_hour() -> None:
    tau = _hours(30)
    src = fi.within_day_source(tau, np.random.default_rng(1))
    assert (tau[src] // DAY == tau // DAY).all()
    assert (src != np.arange(len(tau))).all()


def test_day_block_null_keeps_the_hour() -> None:
    tau = _hours(30)
    src = fi.day_block_source(tau, np.random.default_rng(2))
    assert ((tau[src] // HOUR) % 24 == (tau // HOUR) % 24).all()


def _minutes(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="min", tz="UTC").as_unit("ns")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    v = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
                         "volume": v, "quote_volume": v * c, "trades": rng.integers(5, 50, n),
                         "taker_buy_base": v * rng.uniform(0.3, 0.7, n)}, index=idx)


def _later_changed(m: pd.DataFrame, cut: int) -> pd.DataFrame:
    m2 = m.copy()
    m2.iloc[cut:, m2.columns.get_loc("close")] *= 1.5
    m2.iloc[cut:, m2.columns.get_loc("high")] *= 1.6
    m2.iloc[cut:, m2.columns.get_loc("volume")] *= 7
    m2.iloc[cut:, m2.columns.get_loc("quote_volume")] *= 9
    return m2


def test_bars_read_only_ended_minutes() -> None:
    m = _minutes(6 * 1440)
    cut = 4 * 1440 + 17  # minutes from here on are changed
    m2 = _later_changed(m, cut)
    tau = m.index.asi8[1440:cut + 1:60]  # decisions up to the first changed minute's open
    amount = float(m["quote_volume"].iloc[:1440].sum() / 24 * 2)
    for ids_of, dollar in ((lambda x: fi.time_bar_ids(x, 15), False),
                           (lambda x: fi.time_bar_ids(x, 60), False),
                           (lambda x: fi.dollar_bar_ids(x, amount), True)):
        ch1, e1 = fi.bars_from_ids(m, ids_of(m), fi.minute_sigma(m["close"]), dollar)
        ch2, e2 = fi.bars_from_ids(m2, ids_of(m2), fi.minute_sigma(m2["close"]), dollar)
        w1, w2 = fi.bar_windows(ch1, e1, tau, 24), fi.bar_windows(ch2, e2, tau, 24)
        assert np.allclose(w1, w2)
        assert np.allclose(fi.bar_window_means(ch1, e1, tau, 24),
                           fi.bar_window_means(ch2, e2, tau, 24))
        assert (e1[np.searchsorted(e1, tau, side="right") - 1] <= tau).all()


def test_an_hourly_bar_ending_at_the_decision_is_used() -> None:
    m = _minutes(3 * 1440)
    ch, end = fi.bars_from_ids(m, fi.time_bar_ids(m, 60), fi.minute_sigma(m["close"]), False)
    tau = np.array([m.index.asi8[2 * 1440]])
    last = np.searchsorted(end, tau, side="right") - 1
    assert end[last][0] == tau[0]


def test_minute_summaries_and_size_baseline_read_only_the_past() -> None:
    m = _minutes(40 * 1440, seed=3)
    cut = 35 * 1440
    m2 = _later_changed(m, cut)
    s1, s2 = fi.minute_summaries(m), fi.minute_summaries(m2)
    assert np.allclose(s1[:cut], s2[:cut], equal_nan=True)
    logc = np.log(m["close"].to_numpy()[59::60])
    logc2 = np.log(m2["close"].to_numpy()[59::60])
    tau = m.index.asi8[59::60] + fi.MINUTE
    b1, _ = fi.size_baseline(logc, tau)
    b2, _ = fi.size_baseline(logc2, tau)
    k = cut // 60  # hourly rows whose last minute is before the cut
    assert np.allclose(b1[:k], b2[:k], equal_nan=True)


def test_shuffled_labels_touch_only_the_masked_rows() -> None:
    tau = _hours(60)
    y = np.arange(len(tau), dtype=float)
    mask = tau < tau[0] + 40 * DAY
    out = fi.shuffled_labels(y, mask, tau, "other day", np.random.default_rng(4))
    assert np.array_equal(out[~mask], y[~mask])
    assert sorted(out[mask]) == sorted(y[mask])


def test_triple_barrier_labels_the_first_touch() -> None:
    n = 400
    t = pd.date_range("2022-01-01", periods=n, freq="min", tz="UTC").as_unit("ns").asi8
    close = np.full(n, 100.0)
    close[61:70] = 100 * np.exp(0.02)  # decision at minute 60 (pos 59 is C(t)): up first
    close[130:140] = 100 * np.exp(-0.02)  # decision at minute 120: down first
    pos = np.array([59, 119, 239, 350])
    tau = t[pos] + fi.MINUTE
    width = np.array([0.01, 0.01, 0.01, 0.01])
    lab = fi.triple_barrier(close, t, pos, tau, 1, width)
    assert lab[0] == 1 and lab[1] == -1 and lab[2] == 0
    assert np.isnan(lab[3])  # the window runs past the data
    gap_t = t.copy()
    gap_t[250:] += 5 * fi.MINUTE  # a gap inside the third window
    assert np.isnan(fi.triple_barrier(close, gap_t, pos[2:3], tau[2:3], 1, width[2:3])[0])


def test_the_barrier_width_reads_only_ended_returns() -> None:
    rng = np.random.default_rng(5)
    logc = np.cumsum(rng.normal(0, 0.01, 6000))
    later = logc.copy()
    later[4000:] += 3.0
    a = fi.past_h_sigma(logc, 24, window=1000, min_obs=200)
    b = fi.past_h_sigma(later, 24, window=1000, min_obs=200)
    assert np.allclose(a[:4000], b[:4000], equal_nan=True)  # step 4000 itself is known at 4000


def test_leader_features_read_only_minutes_up_to_the_decision() -> None:
    alt, lead = _minutes(5 * 1440, seed=7), _minutes(5 * 1440, seed=8)
    cut = 4 * 1440 + 3
    alt2, lead2 = _later_changed(alt, cut), _later_changed(lead, cut)
    pos = np.arange(1500, cut, 60)  # decision minutes before the first changed one
    a, names = fi.leader_features(alt, lead, pos, "L")
    b, _ = fi.leader_features(alt2, lead2, pos, "L")
    assert len(names) == a.shape[1] == 18
    assert np.allclose(a, b, equal_nan=True)


def test_the_minute_target_scaling_reads_only_ended_returns() -> None:
    m = _minutes(30 * 1440, seed=9)
    close, open_ns = m["close"].to_numpy(), m.index.asi8
    pos = np.arange(59, len(m) - 20, 60)
    z, past = fi.minute_z_target(close, open_ns, pos, 15, window=200, min_obs=50)
    close2 = close.copy()
    close2[20 * 1440:] *= 1.3
    z2, past2 = fi.minute_z_target(close2, open_ns, pos, 15, window=200, min_obs=50)
    k = np.searchsorted(pos, 20 * 1440 - 16)  # decisions whose next 15 minutes end before
    assert np.allclose(z[:k], z2[:k], equal_nan=True)
    assert np.allclose(past[:k], past2[:k], equal_nan=True)
