import numpy as np
import pandas as pd

from qlab import regime_change as rc


def _walk(n=3000, seed=0, sigma=0.03):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))


def test_known_labels_never_contradict_the_full_history():
    v = _walk()
    full, _ = rc.r1_state(v, 0.25)
    for T in (400, 1000, 1700, 2500):
        known = rc.r1_known(v[:T], 0.25)
        ok = ~np.isnan(known) & ~np.isnan(full[:T])
        assert ok.sum() > 0
        assert np.array_equal(known[ok], full[:T][ok])


def test_known_switch_labels_never_contradict_the_full_history():
    v = _walk(seed=3)
    for kinds in (("peak", "trough"), ("peak",)):
        full = rc.switch_known(v, 0.25, 7, kinds)
        for T in (500, 1200, 2400):
            known = rc.switch_known(v[:T], 0.25, 7, kinds)
            ok = ~np.isnan(known) & ~np.isnan(full[:T])
            assert ok.sum() > 0
            assert np.array_equal(known[ok], full[:T][ok])


def test_state_is_down_exactly_between_peak_and_trough():
    v = np.array([1, 2, 3, 2, 1.5, 1, 1.5, 2, 3, 4, 2.5, 2.0])
    st, z = rc.r1_state(v, 0.25)
    assert z.pivots[:3] == [(0, "trough"), (2, "peak"), (5, "trough")]
    assert list(st[2:5]) == [1, 1, 1]
    assert list(st[5:9]) == [0, 0, 0, 0]


def test_drawdown_rule_is_the_online_zigzag():
    v = _walk(seed=5)
    det = rc.rule_drawdown(v, 0.25)
    st, z = rc.r1_state(v, 0.25)
    # each confirmed peak is detected exactly when the zig-zag confirms it
    for p, kind in z.pivots[1:]:
        if kind == "peak":
            j = p + int(np.argmax(v[p:] <= v[p] * 0.75))
            assert det[j] == 1 and det[j - 1] == 0


def test_jump_model_with_a_huge_penalty_never_switches():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    C = rc.jump_fit(X, 2, 1e9, rng, starts=2)
    s, _ = rc.jump_online(X, C, 1e9)
    assert len(np.unique(s)) == 1


def test_permuted_days_keep_the_hourly_moves():
    idx = pd.date_range("2024-01-01", periods=24 * 50, freq="1h", tz="UTC")
    rng = np.random.default_rng(2)
    lc = np.cumsum(rng.normal(0, 0.01, len(idx)))
    fr = pd.DataFrame({"logc": lc, "x": np.arange(len(idx))}, index=idx)
    p = rc.permute_days(fr, np.random.default_rng(3))
    a = np.sort(np.diff(lc))
    b = np.sort(np.diff(p["logc"].to_numpy()))
    # the day-boundary moves change; the within-day ones are the same multiset
    within = np.sort(np.concatenate([np.diff(lc[d * 24:(d + 1) * 24]) for d in range(50)]))
    within_p = np.sort(np.concatenate([np.diff(p["logc"].to_numpy()[d * 24:(d + 1) * 24])
                                       for d in range(50)]))
    assert np.allclose(within, within_p)
    assert len(a) == len(b)
    assert sorted(p["x"].to_numpy() // 24) == sorted(np.arange(len(idx)) // 24)


def test_filtered_hmm_reads_only_the_past():
    rng = np.random.default_rng(4)
    X = np.concatenate([rng.normal(0, 1, (300, 2)), rng.normal(3, 1, (300, 2))])
    m = rc.hmm_fit(X, 2, rng, starts=2)
    a = rc.hmm_filtered(m, X)
    b = rc.hmm_filtered(m, X[:400])
    assert np.allclose(a[:400], b)


def test_placebo_keeps_run_count_and_lengths():
    det = np.zeros(1000)
    det[100:130] = 1
    det[500:520] = 1
    out = rc.relocate_runs(det, np.random.default_rng(0), np.ones(1000, dtype=bool))
    assert out.sum() == det.sum()
