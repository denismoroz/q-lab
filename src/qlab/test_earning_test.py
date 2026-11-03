"""«Заработок отличим от нуля»: the bootstrap reading and its statistics (docs/NOISE_BAR.md)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from qlab import earning_test
from qlab.cli import app
from qlab.overlay_compare import block_length
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
)
from qlab.rules.loader import DEFAULT_RULES_DIR, load

RULES = load("2026-11-03.1", DEFAULT_RULES_DIR)


def _days(values: np.ndarray) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index)


def _noisy(mean: float, n: int = 400, seed: int = 1, sd: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_a_clear_earning_is_told_from_zero():
    values = _noisy(0.0)
    found = earning_test.bootstrap(values - values.mean() + 0.002, additive=True)
    assert found is not None
    assert found.p <= 0.001  # at most one resample of 2000 reached zero
    assert 0.0 < found.ann_low < 0.73 < found.ann_high  # 0.002 a day is about 73% a year
    assert found.days == 400 and found.block_days == block_length(400)


def test_a_series_with_no_edge_is_not_told_from_zero():
    values = _noisy(0.0)
    found = earning_test.bootstrap(values - values.mean(), additive=True)  # an average day of 0
    assert found is not None
    assert 0.3 < found.p < 0.7
    assert found.ann_low < 0 < found.ann_high


def test_a_losing_series_has_p_near_one():
    found = earning_test.bootstrap(_noisy(-0.002), additive=True)
    assert found.p == 1.0 and found.ann_high < 0


def test_the_same_days_give_the_same_numbers_and_the_seed_is_fixed():
    values = _noisy(0.001)
    a = earning_test.bootstrap(values, additive=True)
    b = earning_test.bootstrap(values, additive=True)
    assert a == b and a.seed == earning_test.SEED == 20261103 and a.n_boot == 2000


def test_the_block_is_the_overlay_comparisons():
    """One rule for the blocks, shared with the comparison of an overlay with its base."""
    assert block_length(1000) == 10 and block_length(8) == 2 and block_length(400) == 7


def test_compounding_and_summing_agree_on_a_small_series_and_differ_on_a_large_one():
    small = _noisy(0.0005, sd=0.002)
    summed = earning_test.bootstrap(small, additive=True)
    compounded = earning_test.bootstrap(small, additive=False)
    assert summed.p == compounded.p  # the same average day
    assert compounded.ann_high > summed.ann_high  # compounding adds interest on interest


def test_too_few_days_or_a_missing_day_is_not_defined():
    assert earning_test.bootstrap(_noisy(0.002, n=earning_test.MIN_DAYS - 1), additive=True) is None
    with_gap = _noisy(0.002)
    with_gap[10] = np.nan
    assert earning_test.bootstrap(with_gap, additive=True) is None
    assert earning_test.metrics_of(_days(_noisy(0.002, n=10)), additive=True) == {}


def test_the_bars_of_a_day_are_summed_or_compounded_into_one_day():
    hourly = pd.Series(0.001, index=pd.date_range("2025-01-01", periods=24 * 60, freq="1h",
                                                  tz="UTC"))
    summed = earning_test.daily_money(hourly, additive=True)
    compounded = earning_test.daily_money(hourly, additive=False)
    assert len(summed) == 60 and summed.iloc[0] == pytest.approx(0.024)
    assert compounded.iloc[0] == pytest.approx(1.001 ** 24 - 1)


def test_the_metrics_a_run_stores():
    m = earning_test.metrics_of(_days(_noisy(0.002)), additive=True)
    assert set(m) == set(earning_test.KEYS)
    assert m["cap_mean_positive_p"] <= 0.001 and m["cap_earning_days"] == 400.0
    assert m["cap_ann_ci_low"] < m["cap_ann_ci_high"]


def test_the_reading_in_plain_words():
    m = {"cap_mean_positive_p": 0.012, "cap_ann_ci_low": 0.031, "cap_ann_ci_high": 0.14,
         "cap_earning_days": 400.0, "cap_earning_block_days": 7.0}
    assert earning_test.read(m).state == earning_test.PASSES
    assert earning_test.line_ru(m) == (
        "заработок отличим от нуля: p = 0.012, 95% интервал дохода в год от +3.1% до +14.0% "
        "(ничего не решает)")
    weak = {**m, "cap_mean_positive_p": 0.3, "cap_ann_ci_low": -0.05}
    assert earning_test.read(weak).state == earning_test.FAILS
    assert "не отличим от нуля: p = 0.300" in earning_test.line_ru(weak)
    assert "от −5.0% до" in earning_test.line_ru(weak)
    assert "меньше 1 из 2000" in earning_test.line_ru({**m, "cap_mean_positive_p": 0.0})
    assert earning_test.read({}).state == earning_test.UNDEFINED
    assert earning_test.line_ru({"ann_return_net": 0.1}) is None
    # The line is p <= 0.05, inclusive.
    assert earning_test.read({**m, "cap_mean_positive_p": 0.05}).distinct is True
    assert earning_test.read({**m, "cap_mean_positive_p": 0.0501}).distinct is False


def test_no_rule_reads_the_reading():
    """It decides nothing: no rule and no informational reading of the latest rules names it."""
    names = {r.metric for r in RULES.rules} | {r.metric for r in RULES.informational}
    assert not names & set(earning_test.KEYS)


# ---- the registry --------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _idea(session, idea_id, *, source="binance"):
    repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
    return repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d",
                         costs_model={}, code_ref="x:y")


def _trial(session, spec, metrics, route, tag="t"):
    return repo.add_trial(
        session, spec_id=spec.id, config_hash=tag, code_sha="x", params={},
        status=TrialStatus.OK, started_at=datetime.now(UTC), metrics=metrics, route=route,
        route_reason="r", source=TrialSource.IMPORTED)


def _m(p, rank, *, low=0.01, high=0.2):
    return {"ann_return_net": 0.1, "cap_mean_positive_p": p, "cap_ann_ci_low": low,
            "cap_ann_ci_high": high, "cap_earning_days": 500.0, "cap_earning_block_days": 8.0,
            "noise_cap_return_percentile": rank}


@pytest.fixture()
def registry(session):
    _trial(session, _idea(session, "good"), _m(0.001, 0.995), TrialRoute.PAPER, "a")
    _trial(session, _idea(session, "lucky"), _m(0.20, 0.995, low=-0.02), TrialRoute.PAPER, "b")
    _trial(session, _idea(session, "edge"), _m(0.02, 0.90), TrialRoute.REJECT, "c")
    _trial(session, _idea(session, "weak", source="hyperliquid"), _m(0.4, 0.5),
           TrialRoute.NEEDS_FORWARD, "d")
    # Waiting for its recomputation: a run with no reading. Left out: a noise book.
    _trial(session, _idea(session, "old"), {"ann_return_net": 0.1}, TrialRoute.PAPER, "e")
    _trial(session, _idea(session, "noise-001"), _m(0.001, 0.5), TrialRoute.REJECT, "f")
    session.commit()
    return session


def test_the_latest_run_of_each_idea_is_counted_and_the_waiting_ones_named(registry):
    rows, missing = earning_test.collect(registry, RULES)
    assert [r.idea_id for r in rows] == ["edge", "good", "lucky", "weak"]
    assert missing == 1
    by_id = {r.idea_id: r for r in rows}
    assert by_id["good"].noise_passes is True and by_id["edge"].noise_passes is False
    assert by_id["weak"].stage == "other"


def test_how_many_would_pass_and_fail_at_the_line(registry):
    rows, _ = earning_test.collect(registry, RULES)
    summary = earning_test.summarize(rows)
    assert summary["total"] == {"passes": 2, "fails": 2}
    assert summary["by_outcome"]["passed"] == {"passes": 1, "fails": 1}
    assert summary["by_outcome"]["rejected"] == {"passes": 1, "fails": 0}
    assert summary["against_noise_bar"]["noise passes"] == {"passes": 1, "fails": 1}
    assert summary["against_noise_bar"]["noise does not pass"] == {"passes": 1, "fails": 1}
    text = "\n".join(earning_test.render_summary(summary))
    assert "at p <= 0.05" in text and "noise passes" in text
    assert "edge" in "\n".join(earning_test.render_table(rows))


def test_the_morning_report_section(registry):
    lines = earning_test.report_lines(registry, RULES)
    text = "\n".join(lines)
    assert "ничего не решает" in text and "ещё 1 ждут пересчёта" in text
    assert "отличимы от нуля 2, не отличимы 2" in text
    assert "пропускает планка шума: отличимы от нуля 1, не отличимы 1" in text
    assert "qlab earning-test" in text


def test_a_registry_with_no_reading_says_what_it_waits_for(session):
    _trial(session, _idea(session, "old"), {"ann_return_net": 0.1}, TrialRoute.PAPER, "e")
    session.commit()
    (line,) = earning_test.report_lines(session, RULES)
    assert "Замера пока нет ни у одной из 1 идей" in line


def test_the_command_prints_the_table_and_the_totals(registry, monkeypatch):
    class _Scope:
        def __enter__(self):
            return registry

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("qlab.cli.session_scope", lambda: _Scope())
    result = CliRunner().invoke(app, ["earning-test"])
    assert result.exit_code == 0, result.output
    assert "latest judged run of each idea" in result.output
    assert "1 idea(s) have a run" in result.output
    assert "lucky" in result.output and "noise passes" in result.output
    as_json = CliRunner().invoke(app, ["earning-test", "--json"])
    assert as_json.exit_code == 0
    import json

    payload = json.loads(as_json.output)
    assert payload["summary"]["total"] == {"passes": 2, "fails": 2}
    assert payload["without_reading"] == 1
