"""Tests for `qlab.overlay_compare`: did the overlay help, on the same days, and is the
difference distinguishable from chance (docs/OVERLAYS.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qlab.registry.db as db_module
from qlab import overlay_compare as oc
from qlab.registry import repo
from qlab.registry.deflation import family_trials as deflation_family
from qlab.registry.models import (
    AssetClass,
    Base,
    DataSnapshot,
    Idea,
    Profile,
    SourceType,
    Spec,
    TrialRoute,
    TrialStatus,
)


def days(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC", name=None)


def noisy(n: int, seed: int, mean: float = 0.0005, sd: float = 0.02) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n), index=days(n))


# ---- the numbers of one series ----------------------------------------------------------------


def test_window_metrics_on_a_known_series() -> None:
    daily = pd.Series([0.10, -0.05, 0.10, -0.05] * 10, index=days(40))
    m = oc.window_metrics(daily)
    growth = (1.10 * 0.95) ** 20
    assert m["days"] == 40 and m["total_return"] == pytest.approx(growth - 1)
    assert m["ann_return"] == pytest.approx(growth ** (365.25 / 40) - 1)
    assert m["max_dd"] == pytest.approx(-0.05)           # the worst fall from a peak
    assert m["sharpe"] == pytest.approx(
        daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))


def test_a_flat_series_has_a_zero_sharpe_not_a_crash() -> None:
    assert oc.window_metrics(pd.Series(0.0, index=days(40)))["sharpe"] == 0.0


# ---- the bootstrap -----------------------------------------------------------------------------


def test_identical_series_differ_by_nothing() -> None:
    base = noisy(400, 1)
    result = oc.bootstrap_difference(base, base.copy(), n_boot=300)
    assert result["delta_sharpe_ci"] == [0.0, 0.0] and result["delta_sharpe_p"] == 1.0
    assert result["block_days"] == 7                       # round(400 ** (1/3))


def test_a_clear_improvement_is_distinguishable_and_noise_is_not() -> None:
    base = noisy(900, 2)
    better = base + 0.004                                   # +0.4% every day, same noise
    result = oc.bootstrap_difference(base, better, n_boot=500)
    low, high = result["delta_sharpe_ci"]
    assert 0 < low < high and result["delta_sharpe_p"] < 0.01
    assert result["mean_daily_diff_ann_ci"][0] > 0
    other = noisy(900, 99)                                  # another strategy with the same drift
    result = oc.bootstrap_difference(base, other, n_boot=500)
    assert result["delta_sharpe_ci"][0] < 0 < result["delta_sharpe_ci"][1]
    assert result["delta_sharpe_p"] > 0.05


def test_the_bootstrap_is_reproducible_and_pairs_the_days() -> None:
    base, variant = noisy(300, 3), noisy(300, 4)
    a = oc.bootstrap_difference(base, variant, n_boot=200)
    b = oc.bootstrap_difference(base, variant, n_boot=200)
    assert a == b
    # The same days are drawn for both series: a variant that IS the base plus a constant has a
    # difference of the mean only, so its interval is narrow compared with two unpaired draws.
    shifted = base + 0.001
    paired = oc.bootstrap_difference(base, shifted, n_boot=200)
    assert paired["mean_daily_diff_ann_ci"][1] - paired["mean_daily_diff_ann_ci"][0] < 1e-9
    with pytest.raises(ValueError, match="same days"):
        oc.bootstrap_difference(base, variant.iloc[1:], n_boot=10)


def test_blocks_keep_neighbouring_days_together() -> None:
    rng = np.random.default_rng(0)
    idx = oc._block_indices(100, 5, 50, rng)
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    steps = np.diff(idx[:, :5], axis=1)
    assert ((steps == 1) | (steps == -99)).all()            # inside a block: consecutive days


def test_compare_series_uses_the_days_both_have() -> None:
    base = noisy(400, 5)
    variant = noisy(400, 6).iloc[100:]
    record = oc.compare_series(base, variant, n_boot=100)
    assert record["days"] == 300 and record["start"] == variant.index[0].date().isoformat()
    assert record["base"]["days"] == record["variant"]["days"] == 300
    assert record["delta_sharpe"] == pytest.approx(
        record["variant"]["sharpe"] - record["base"]["sharpe"])
    with pytest.raises(ValueError, match="nothing to compare"):
        oc.compare_series(base.iloc[:10], variant, n_boot=10)


# ---- the verdict -------------------------------------------------------------------------------


def _record(delta: float, p: float, days_: int = 800) -> dict:
    return {"days": days_, "delta_sharpe": delta, "delta_sharpe_p": p}


def test_the_three_answers() -> None:
    assert oc.verdict(_record(0.4, 0.001), 1) == oc.HELPED
    assert oc.verdict(_record(-0.4, 0.001), 1) == oc.NOT_HELPED
    assert oc.verdict(_record(0.4, 0.2), 1) == oc.UNDISTINGUISHABLE
    assert oc.VERDICT_RU == {"helped": "помогла", "not-helped": "не помогла",
                             "indistinguishable": "неотличимо от случайности"}


def test_many_variants_make_a_pass_harder() -> None:
    record = _record(0.4, 0.02)
    assert oc.verdict(record, 1) == oc.HELPED
    assert oc.verdict(record, 15) == oc.UNDISTINGUISHABLE   # 0.02 * 15 = 0.30
    assert oc.corrected_p(record, 15) == pytest.approx(0.30)
    assert oc.corrected_p(_record(0.4, 0.5), 15) == 1.0     # a probability stays one


def test_a_window_shorter_than_a_year_cannot_tell() -> None:
    assert oc.verdict(_record(0.9, 0.0001, days_=200), 1) == oc.UNDISTINGUISHABLE


# ---- files and the family ----------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    yield db_module
    db_module._engine = None
    db_module._SessionLocal = None


def _idea(session, idea_id: str, parent: str | None = None, driver: str | None = None) -> None:
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, driver_id=driver)
    session.flush()
    session.get(Idea, idea_id).parent_id = parent


def _trial(session, idea_id: str, route: str = "needs-forward"):
    spec = session.query(Spec).filter_by(idea_id=idea_id).first() or repo.add_spec(
        session, idea_id=idea_id, version=1, params={}, data_requirements={}, rebalance="1d",
        costs_model={}, code_ref="a:b")
    from datetime import date

    if session.get(DataSnapshot, "snap") is None:
        repo.add_data_snapshot(session, id="snap", source="binance", instruments={},
                               range_start=date(2020, 1, 1), range_end=date(2026, 1, 1),
                               path="x", rows=1)
    return repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="c", params={},
                          status=TrialStatus.OK, snapshot_id="snap", metrics={},
                          route=TrialRoute(route), route_reason="r")


def test_a_record_is_stored_and_read_back(tmp_path) -> None:
    record = oc.compare_series(noisy(400, 7), noisy(400, 8), n_boot=50)
    record.update(base_id="b", variant_id="b-ov-own")
    path = oc.save_record(record, tmp_path)
    assert path == tmp_path / "compare" / "b-ov-own.json"
    assert oc.load_record("b-ov-own", tmp_path)["days"] == 400
    assert oc.load_record("b-ov-none", tmp_path) is None


def test_every_variant_counts_toward_the_bases_family_in_deflation(registry) -> None:
    from qlab.overlay_stage import ensure_overlay_idea
    from qlab.pipeline.spec import StrategySpec

    def spec_of(idea_id: str) -> StrategySpec:
        return StrategySpec.model_validate({
            "idea_id": idea_id, "title": idea_id, "code_ref": "a:b", "params": {},
            "data": {"source": "binance", "interval": "1d", "start": "2022-01-01",
                     "end": "2023-01-01"},
            "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
            "unexpressed_mechanisms": []})

    with registry.session_scope() as s:
        repo.upsert_driver(s, id="momentum", title="momentum", description="d",
                           kill_condition="k", observable="o")
        _idea(s, "base", driver="momentum")
        _trial(s, "base")
    for name in ("base-ov-own", "base-ov-vol", "base-ov-own-enter"):
        ensure_overlay_idea(spec_of(name), "base")
        with registry.session_scope() as s:
            _trial(s, name)
            _trial(s, name)                                 # a re-run is a trial as well
    with registry.session_scope() as s:
        assert s.get(Idea, "base-ov-own").parent_id == "base"
        assert s.get(Idea, "base-ov-own").driver_id == "momentum"
        count = deflation_family(s, "base", scope="driver")
        assert count.n_trials == 1 + 3 * 2                  # the base's own and every variant's
        assert oc.family_trials(s, "base") == 7
        assert oc.variants_of(s, "base") == ["base-ov-own", "base-ov-own-enter", "base-ov-vol"]


def test_the_table_reads_the_stored_comparisons_at_the_current_family_size(registry, tmp_path):
    with registry.session_scope() as s:
        _idea(s, "base")
        _trial(s, "base")
        for name in ("base-ov-own", "base-ov-vol", "base-ov-enter"):
            _idea(s, name, parent="base")
            _trial(s, name)
        _idea(s, "other")
        _idea(s, "other-ov-own", parent="other")           # another base's: not counted
    better = oc.compare_series(noisy(900, 2), noisy(900, 2) + 0.004, n_boot=200)
    same = oc.compare_series(noisy(900, 2), noisy(900, 11), n_boot=200)
    for name, record in (("base-ov-own", better), ("base-ov-vol", same)):
        oc.save_record({**record, "base_id": "base", "variant_id": name}, tmp_path)
    with registry.session_scope() as s:
        table = oc.table(s, "base", tmp_path)
    assert table["family_size"] == 2                        # only the compared ones
    by_id = {row["idea_id"]: row for row in table["variants"]}
    assert by_id["base-ov-own"]["comparison"]["verdict"] == "helped"
    assert by_id["base-ov-own"]["comparison"]["verdict_ru"] == "помогла"
    assert by_id["base-ov-vol"]["comparison"]["verdict"] == "indistinguishable"
    assert by_id["base-ov-enter"]["comparison"] is None     # run, not compared yet
    assert table["counts"] == {"helped": 1, "not-helped": 0, "indistinguishable": 1}
    assert table["family_trials"] == 4
