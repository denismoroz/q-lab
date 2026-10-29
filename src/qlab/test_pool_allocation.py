"""Tests for `qlab.pool_allocation`: the capital split between strategies (code proposes)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab import paper_rule, pool_allocation
from qlab.pool_allocation import Evidence, MoneyError, Strategy
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialStatus,
)

HOUR_MS = 3_600_000
CONFIG = {
    "version": "test", "method": "inverse_volatility", "min_evidence_days": 28,
    "deployable_capital_usd": 3000, "capital_ladder_usd": [],
    "short_evidence": {"live": "hold_current", "other": "no_weight"},
    "on_decay_alert": "report_only",
}


def _returns(sigma: float, days: int = 30, start: date = date(2026, 9, 1)) -> pd.Series:
    """Daily returns with EXACTLY the sample std `sigma` (ddof 1) and mean zero, alternating."""
    n = days
    base = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    if n % 2:
        base = base - base.mean()
    base = base / np.std(base, ddof=1) * sigma
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="D")
    return pd.Series(base, index=index)


def _strategy(root: str, sigma: float, *, minimum: float | None = 50.0, days: int = 30,
              role: str = "paper", kind: str = "paper", current: float | None = None,
              **kw) -> Strategy:
    evidence = Evidence(kind, _returns(sigma, days), "test") if days else None
    return Strategy(root=root, idea_id=root, label=root.upper(), role=role, evidence=evidence,
                    min_capital_usd=minimum, current_capital_usd=current, **kw)


def _by_root(proposal) -> dict:
    return {ln.root: ln for ln in proposal.lines}


# --- the method, against hand-computed cases ------------------------------------------------


def test_inverse_volatility_weights_are_proportional_to_one_over_sigma() -> None:
    # 1/0.01 = 100, 1/0.02 = 50, 1/0.04 = 25; total 175 -> 4/7, 2/7, 1/7.
    weights = pool_allocation.inverse_volatility({"a": 0.01, "b": 0.02, "c": 0.04})
    assert weights == pytest.approx({"a": 4 / 7, "b": 2 / 7, "c": 1 / 7})
    assert sum(weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pool_allocation.inverse_volatility({"a": 0.0})


def test_daily_std_is_the_sample_std() -> None:
    # mean 0.01, deviations 0, -0.02, 0.02, 0 -> sum of squares 0.0008, / 3, square root
    returns = pd.Series([0.01, -0.01, 0.03, 0.01])
    assert pool_allocation.daily_std(returns) == pytest.approx((0.0008 / 3) ** 0.5)
    assert pool_allocation.daily_std(pd.Series([0.01])) is None


def test_with_equal_correlations_inverse_volatility_is_equal_risk_contribution() -> None:
    """Maillard-Roncalli-Teiletche (2010): at a constant correlation the weights 1/sigma make
    every strategy's risk contribution w_i * (cov w)_i the same."""
    sigma = np.array([0.01, 0.02, 0.05])
    rho = 0.3
    corr = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    cov = corr * np.outer(sigma, sigma)
    w = np.array(list(pool_allocation.inverse_volatility(dict(enumerate(sigma))).values()))
    contributions = w * (cov @ w)
    assert contributions == pytest.approx(np.full(3, contributions.mean()))


def test_a_calmer_strategy_gets_more_and_the_dollars_sum_to_the_pool() -> None:
    result = pool_allocation.allocate(
        [_strategy("calm", 0.01), _strategy("wild", 0.02)], capital=3000, max_count=6,
        config=CONFIG)
    lines = _by_root(result)
    assert lines["calm"].dollars == pytest.approx(2000.0)
    assert lines["wild"].dollars == pytest.approx(1000.0)
    assert lines["calm"].share == pytest.approx(2 / 3)
    assert sum(ln.dollars for ln in result.lines) == pytest.approx(3000.0)
    # the comparison: equal capital would hand the wild one 2/3 of the risk
    assert lines["calm"].equal_dollars == pytest.approx(1500.0)
    assert lines["wild"].equal_risk_share == pytest.approx(2 / 3)
    assert lines["wild"].risk_share == pytest.approx(0.5)


# --- constraints --------------------------------------------------------------------------


def test_a_strategy_whose_minimum_does_not_fit_gets_zero_and_the_rest_is_re_split() -> None:
    # equal volatility -> 500 each of a pool of 1000, but b needs 600: b is out, a takes 1000
    result = pool_allocation.allocate(
        [_strategy("a", 0.01, minimum=100), _strategy("b", 0.01, minimum=600)],
        capital=1000, max_count=6, config=CONFIG)
    lines = _by_root(result)
    assert lines["b"].status == "zero" and lines["b"].dollars == 0.0
    assert "досталось бы $500" in lines["b"].reason and "$600" in lines["b"].reason
    assert lines["a"].dollars == pytest.approx(1000.0)
    # b fits when both stand at half the money: max(100 / 0.5, 600 / 0.5) = 1200
    assert lines["b"].fits_from_usd == pytest.approx(1200.0)
    assert "$1 200" in lines["b"].reason
    again = pool_allocation.allocate(
        [_strategy("a", 0.01, minimum=100), _strategy("b", 0.01, minimum=600)],
        capital=1200, max_count=6, config=CONFIG)
    assert _by_root(again)["b"].dollars == pytest.approx(600.0)


def test_the_lowest_ratio_to_its_minimum_is_dropped_first() -> None:
    # three equal strategies of a pool of 900 (300 each): minima 200, 400, 800 -> the 800 goes
    # first; then 450 each and the 400 fits.
    result = pool_allocation.allocate(
        [_strategy("x", 0.01, minimum=200), _strategy("y", 0.01, minimum=400),
         _strategy("z", 0.01, minimum=800)], capital=900, max_count=6, config=CONFIG)
    lines = _by_root(result)
    assert lines["z"].status == "zero"
    assert lines["x"].dollars == pytest.approx(450.0) and lines["y"].dollars == pytest.approx(450.0)


def test_nobody_fits_means_nobody_is_funded() -> None:
    result = pool_allocation.allocate([_strategy("a", 0.01, minimum=5000)], capital=1000,
                                      max_count=6, config=CONFIG)
    assert _by_root(result)["a"].dollars == 0.0 and "$5 000" in _by_root(result)["a"].reason


def test_an_unknown_minimum_means_no_weight() -> None:
    result = pool_allocation.allocate([_strategy("a", 0.01, minimum=None)], capital=1000,
                                      max_count=6, config=CONFIG)
    assert _by_root(result)["a"].status == "zero" and "не измерил" in _by_root(result)["a"].reason


def test_more_candidates_than_the_cap_is_the_owners_choice() -> None:
    strategies = [_strategy(r, 0.01) for r in ("a", "b", "c")]
    result = pool_allocation.allocate(strategies, capital=3000, max_count=2, config=CONFIG)
    assert all(ln.dollars == 0.0 for ln in result.lines)
    assert result.needs_decision and "лимит" in result.needs_decision[0]


# --- evidence ----------------------------------------------------------------------------


def test_a_strategy_whose_only_evidence_is_the_fit_period_gets_no_weight() -> None:
    fit = _strategy("fitted", 0.001, kind="in_sample", role="stand", minimum=10)
    honest = _strategy("honest", 0.02)
    result = pool_allocation.allocate([fit, honest], capital=3000, max_count=6, config=CONFIG)
    lines = _by_root(result)
    assert lines["fitted"].dollars == 0.0 and lines["fitted"].status == "zero"
    assert "периоде подбора" in lines["fitted"].reason
    # the calmest series in the pool, and still nothing: everything goes to the honest one
    assert lines["honest"].dollars == pytest.approx(3000.0)


def test_stand_forward_evidence_can_carry_a_weight_and_is_labelled() -> None:
    result = pool_allocation.allocate(
        [_strategy("fwd", 0.01, kind="stand_forward", role="stand")], capital=1000,
        max_count=6, config=CONFIG)
    line = _by_root(result)["fwd"]
    assert line.status == "weighted" and line.evidence_kind == "stand_forward"
    assert "проверка вперёд на стенде" in "\n".join(pool_allocation.report_lines(result, []))


def test_too_few_days_paper_gets_nothing_and_a_date() -> None:
    result = pool_allocation.allocate(
        [_strategy("young", 0.01, days=10)], capital=1000, max_count=6, config=CONFIG)
    line = _by_root(result)["young"]
    assert line.status == "zero" and "10 из 28" in line.reason
    # its last day is 2026-09-10; 18 more days -> 2026-09-28
    assert line.ready_on == "2026-09-28"


def test_a_live_strategy_with_too_few_days_keeps_its_money_and_the_rest_is_split() -> None:
    live = _strategy("live", 0.001, days=15, role="live", kind="live", current=90.0, minimum=92)
    paper = _strategy("paper", 0.02)
    result = pool_allocation.allocate([live, paper], capital=3000, max_count=6, config=CONFIG)
    lines = _by_root(result)
    assert lines["live"].status == "held" and lines["live"].dollars == 90.0
    assert lines["paper"].dollars == pytest.approx(2910.0)
    assert result.held == 90.0 and result.pool == pytest.approx(2910.0)
    assert sum(ln.dollars for ln in result.lines) == pytest.approx(3000.0)


def test_with_enough_days_a_live_strategy_is_weighed_like_the_others() -> None:
    live = _strategy("live", 0.01, days=30, role="live", kind="live", current=90.0, minimum=92)
    paper = _strategy("paper", 0.01)
    result = pool_allocation.allocate([live, paper], capital=3000, max_count=6, config=CONFIG)
    assert _by_root(result)["live"].status == "weighted"
    assert _by_root(result)["live"].dollars == pytest.approx(1500.0)


def test_a_decay_flag_changes_nothing_until_the_owner_says_to_exclude() -> None:
    flagged = _strategy("flagged", 0.01, decay_alert="below")
    other = _strategy("other", 0.01)
    shown = pool_allocation.allocate([flagged, other], capital=1000, max_count=6, config=CONFIG)
    assert _by_root(shown)["flagged"].dollars == pytest.approx(500.0)
    excluding = {**CONFIG, "on_decay_alert": "exclude"}
    cut = pool_allocation.allocate([flagged, other], capital=1000, max_count=6, config=excluding)
    assert _by_root(cut)["flagged"].dollars == 0.0
    assert _by_root(cut)["other"].dollars == pytest.approx(1000.0)


def test_correlations_use_the_common_window_and_hide_short_ones() -> None:
    a = _strategy("a", 0.01, days=30)
    b = _strategy("b", 0.01, days=30)
    b.evidence = Evidence("paper", -b.evidence.returns, "negated")
    short = _strategy("c", 0.01, days=30)
    short.evidence = Evidence("paper", short.evidence.returns.iloc[-10:], "ten days")
    result = pool_allocation.allocate([a, b, short], capital=3000, max_count=6, config=CONFIG)
    assert len(result.correlations) == 1
    pair = result.correlations[0]
    assert (pair["a"], pair["b"], pair["n"]) == ("a", "b", 30)
    assert pair["correlation"] == pytest.approx(-1.0)
    assert "пар с общим окном короче 28 дней: 2" in result.correlation_note


# --- the strategy-money rule -----------------------------------------------------------------


def test_a_book_without_strategy_capital_is_refused() -> None:
    with pytest.raises(MoneyError, match="strategy_capital_usd"):
        pool_allocation.assert_strategy_money("x", {"paper_file": "a.csv"})
    pool_allocation.assert_strategy_money("x", {"strategy_capital_usd": 257})


def test_a_live_book_on_account_equity_is_refused() -> None:
    with pytest.raises(MoneyError, match="account"):
        pool_allocation.assert_strategy_money(
            "frab", {"strategy_capital_usd": 90, "live": True, "equity_from": "total_equity"})
    pool_allocation.assert_strategy_money(
        "frab", {"strategy_capital_usd": 90, "live": True, "equity_from": "strategy_pnl"})


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _export(folder: Path, name: str, first: date, days: int, *, jump_before: date | None = None
            ) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    ts = [_ms(first) + h * HOUR_MS for h in range(24 * days + 1)]
    equity = [100.0 * 1.0005 ** (i / 24) for i in range(len(ts))]
    if jump_before is not None:  # history the book replayed at launch: a wild move
        for i, t in enumerate(ts):
            if t < _ms(jump_before):
                equity[i] = 50.0 + (i % 7) * 20.0
    pd.DataFrame({"strategy_id": 5, "ts_ms": ts, "equity": equity}).to_csv(
        folder / name, index=False)


def test_only_the_realised_part_of_a_book_is_used(tmp_path: Path) -> None:
    _export(tmp_path / "2026-10-25", "t.csv", date(2026, 9, 1), 40, jump_before=date(2026, 10, 1))
    cfg = {"paper_file": "t.csv", "strategy_id": 5, "strategy_capital_usd": 100,
           "realised_from": "2026-10-01T00:00:00+00:00"}
    evidence = pool_allocation.book_evidence("t", cfg, root=tmp_path)
    assert evidence.kind == "paper"
    assert evidence.first == date(2026, 10, 1)  # the first realised row closes 09-30
    # a smooth 0.05% a day: the wild replayed history never reaches the volatility
    assert pool_allocation.daily_std(evidence.returns) == pytest.approx(0.0, abs=1e-9)
    whole = pool_allocation.book_evidence("t", {**cfg, "realised_from": None}, root=tmp_path)
    assert pool_allocation.daily_std(whole.returns) > 0.1


def test_daily_returns_keep_complete_days_only() -> None:
    index = pd.date_range("2026-10-01 00:00", "2026-10-03 12:00", freq="h", tz="UTC")
    day = index.normalize()
    equity = pd.Series(np.where(day == pd.Timestamp("2026-10-03", tz="UTC"), 130.0,
                                np.where(day == pd.Timestamp("2026-10-02", tz="UTC"),
                                         110.0, 100.0)), index=index)
    returns = pool_allocation.daily_returns(equity)
    # the 00:00 row of a day closes the day before: 09-30 closes at 100, 10-01 at 110, 10-02 at
    # 130. The partial 10-03 is dropped, so two returns are left.
    assert list(returns.index.strftime("%m-%d")) == ["10-01", "10-02"]
    assert list(returns) == pytest.approx([110 / 100 - 1, 130 / 110 - 1])


# --- changes since the last proposal -----------------------------------------------------------


def test_triggers_fire_on_events_not_on_drift() -> None:
    first = pool_allocation.allocate([_strategy("a", 0.01), _strategy("b", 0.01, days=10)],
                                     capital=1000, max_count=6, config=CONFIG)
    snap = pool_allocation.snapshot(first)
    assert pool_allocation.changes_since(None, first)[0].startswith("Это первый расчёт")
    # same strategies, different volatility: weights move, nothing fires
    drift = pool_allocation.allocate([_strategy("a", 0.02), _strategy("b", 0.01, days=10)],
                                     capital=1000, max_count=6, config=CONFIG)
    assert pool_allocation.changes_since(snap, drift) == [
        "С прошлой ночи ничего из списка поводов не случилось."]
    # b reaches the minimum days, c appears, a flag and a ready paper book
    grown = pool_allocation.allocate(
        [_strategy("a", 0.01, decay_alert="below", paper_ready=True), _strategy("b", 0.01),
         _strategy("c", 0.01)], capital=1000, max_count=6, config=CONFIG)
    text = "\n".join(pool_allocation.changes_since(snap, grown))
    assert "B: набрала 28 дней" in text and "C: появилась в пуле" in text
    assert "A: детектор затухания" in text and "A: по правилу бумаги" in text
    assert "ушла из пула" in "\n".join(pool_allocation.changes_since(snap, pool_allocation.allocate(
        [_strategy("a", 0.01)], capital=1000, max_count=6, config=CONFIG)))


# --- the config and the report -----------------------------------------------------------------


def test_the_shipped_config_loads_and_uses_the_owners_four_weeks() -> None:
    config = pool_allocation.load_config(Path("night/allocation.yaml"))
    assert config["method"] == "inverse_volatility"
    assert config["min_evidence_days"] == paper_rule.MIN_REPLAY_DAYS
    assert config["on_decay_alert"] == "report_only"
    assert pool_allocation.max_strategies(config) == 6


def test_the_shipped_config_splits_nothing_each_strategy_has_its_own_money() -> None:
    # owner, 2026-10-29: «нет. у каждой стратегии свои настройки»
    assert pool_allocation.load_config(Path("night/allocation.yaml"))["split"] is False


def test_without_a_split_each_strategy_is_shown_against_its_own_money() -> None:
    enough = _strategy("bv2", 0.005, minimum=150.0, own_capital_usd=257.0, decay_percentile=72.0)
    short = _strategy("trend", 0.004, minimum=7500.0, own_capital_usd=3000.0, paper_ready=True)
    stand = _strategy("xsmom", 0.02, minimum=240.0, role="stand", days=0)
    text = "\n".join(pool_allocation.own_settings_lines([enough, short, stand]))
    assert "деньги в настройках $257" in text and "Своих денег хватает." in text
    assert "**Своих денег не хватает**: нужно от $7 500" in text
    assert "книги у неё нет" in text and "72 из 100" in text and "готова" in text
    for pooled in ("Предложено", "от всей суммы", "общей сумме", "$3 000 — рабочее"):
        assert pooled not in text


def test_the_night_without_a_split_names_its_own_section(session, tmp_path: Path) -> None:
    out = pool_allocation.run_allocation(
        session, date(2026, 10, 25), date(2026, 10, 24), books={},
        config={**CONFIG, "split": False}, root=tmp_path / "paper", night_dir=tmp_path / "nights")
    assert out["title"] == pool_allocation.OWN_SETTINGS_TITLE and out["proposal"] is None


def test_an_unknown_method_is_refused_not_replaced(tmp_path: Path) -> None:
    text = Path("night/allocation.yaml").read_text(encoding="utf-8").replace(
        "method: inverse_volatility", "method: risk_parity_optimised")
    path = tmp_path / "a.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="method"):
        pool_allocation.load_config(path)


def test_the_report_is_plain_russian_and_asks_the_owner() -> None:
    result = pool_allocation.allocate(
        [_strategy("calm", 0.01), _strategy("young", 0.02, days=5)], capital=3000, max_count=6,
        config=CONFIG)
    text = "\n".join(pool_allocation.report_lines(result, ["Первый расчёт."]))
    assert "решение и перевод денег — за владельцем" in text
    assert "Нужно решение владельца" in text
    assert "Предложено $3 000" in text and "5 из 28" in text


# --- end to end on a small registry --------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id, *, status=IdeaStatus.PAPER, parent=None, minimum=None):
    idea = repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                            asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                            status=status)
    idea.parent_id = parent
    session.flush()
    if minimum is not None:
        spec = repo.add_spec(session, idea_id=idea_id, version=1, params={}, data_requirements={},
                             rebalance="1h", costs_model={}, code_ref="a:b")
        repo.add_data_snapshot(session, id=f"snap-{idea_id}", source="x", instruments={},
                               range_start=date(2025, 1, 1), range_end=date(2026, 10, 1),
                               path="x", rows=1)
        repo.add_trial(session, spec_id=spec.id, config_hash="h", code_sha="x", params={},
                       status=TrialStatus.OK, metrics={"min_capital_usd": minimum},
                       route=TrialRoute.PAPER, snapshot_id=f"snap-{idea_id}")


def test_variants_of_one_strategy_are_one_strategy_and_the_run_writes_nothing(
        session, tmp_path: Path) -> None:
    _idea(session, "root", minimum=99.0)
    _idea(session, "variant", parent="root", minimum=128.0)
    _idea(session, "other", minimum=2000.0)
    export = tmp_path / "paper" / "2026-10-25"
    for name, sid in (("r.csv", 1), ("v.csv", 2), ("o.csv", 3)):
        _export(export, name, date(2026, 9, 1), 40)
        frame = pd.read_csv(export / name)
        frame["strategy_id"] = sid
        # a little noise so the volatility is positive
        frame["equity"] = frame["equity"] * (1 + 0.002 * np.sin(np.arange(len(frame)) * sid))
        frame.to_csv(export / name, index=False)
    books = {
        "root": {"book": "r", "label": "Root", "paper_file": "r.csv", "strategy_id": 1,
                 "strategy_capital_usd": 100, "live": False},
        "variant": {"book": "v", "label": "Variant", "paper_file": "v.csv", "strategy_id": 2,
                    "strategy_capital_usd": 100, "live": False},
        "other": {"book": "o", "label": "Other", "paper_file": "o.csv", "strategy_id": 3,
                  "strategy_capital_usd": 100, "live": False},
    }
    before = session.execute(__import__("sqlalchemy").text("select count(*) from trial")).scalar()

    out = pool_allocation.run_allocation(
        session, date(2026, 10, 25), date(2026, 10, 24), books=books, config=CONFIG,
        root=tmp_path / "paper", night_dir=tmp_path / "nights")

    roots = [ln["root"] for ln in out["proposal"]["lines"]]
    assert roots == ["root", "other"]  # the variant is not a strategy of its own
    by = {ln["root"]: ln for ln in out["proposal"]["lines"]}
    assert by["root"]["other_books"] == ["Variant"]
    assert by["root"]["min_capital_usd"] == 99.0  # the representative's own measurement
    assert by["other"]["status"] == "zero" and "2 000" in by["other"]["reason"]
    assert json.dumps(out)  # saved into the night's state as JSON
    after = session.execute(__import__("sqlalchemy").text("select count(*) from trial")).scalar()
    assert before == after  # no trial, no status: a proposal only


def test_the_report_warns_when_a_held_strategy_would_take_most_of_the_pool() -> None:
    live = _strategy("quiet", 0.0005, days=15, role="live", kind="live", current=90.0, minimum=92)
    paper = _strategy("noisy", 0.01)
    result = pool_allocation.allocate([live, paper], capital=3000, max_count=6, config=CONFIG)
    text = "\n".join(pool_allocation.report_lines(result, []))
    assert "Потолок на стратегию особенно важен для «QUIET»" in text and "95%" in text


def test_the_morning_report_carries_the_section_and_a_failed_stage_is_a_line(
        monkeypatch, tmp_path: Path) -> None:
    from qlab import night

    monkeypatch.setattr(night, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    state = {"results": [], "allocation": {"lines": ["Код ничего не двигает."], "proposal": None,
                                           "snapshot": {}, "changes": []}}
    report = night.write_report(date(2026, 10, 25), state).read_text(encoding="utf-8")
    assert f"## {pool_allocation.SECTION_TITLE}" in report and "Код ничего не двигает." in report

    def boom(*args, **kwargs):
        raise RuntimeError("no panel")

    broken: dict = {}
    monkeypatch.setattr(pool_allocation, "night_signals", lambda *a, **k: ([], []))
    monkeypatch.setattr(pool_allocation, "run_allocation", boom)
    pool_allocation.night_stage(date(2026, 10, 25), date(2026, 10, 24), broken,
                                root=tmp_path / "none")
    assert broken["allocation"]["lines"] == ["не посчитано: RuntimeError: no panel"]


def test_each_strategy_says_from_what_sum_it_works() -> None:
    """Owner, 2026-10-26: «на каждую стратегию выводить минимум возможный - чтобы стратегия
    работала и будем отталкиваться от него. минимум должен быть обоснованный»."""
    from qlab.capital_floor import Floor, Step
    from qlab.pool_allocation import Line, Proposal, start_lines, working_minimum

    def step(usd, below=0.0, corr=0.99, net=0.05):
        return Step(usd=usd, dropped_positions=0.0, below_minimum=below, correlation=corr,
                    net_return=net, works=below == 0 and corr >= 0.8 and net > 0)

    bv2 = Floor(idea_id="bv2", floor_usd=150.0, position_floor_usd=99.0, money_in_params=True,
                window=("2026-03-09", "2026-09-30"),
                steps=[step(100, below=0.02), step(150), step(10_000, corr=1.0)], note="")
    weights_only = Floor(idea_id="w", floor_usd=640.0, position_floor_usd=640.0,
                         money_in_params=False, window=("2025-01-01", "2026-09-01"), steps=[],
                         note="")

    def line(label, role, floor, ready=None, current=None):
        return Line(root=label, idea_id=label, label=label, role=role, status="zero",
                    evidence_kind=None, days=0, first=None, last=None, sigma=None, dollars=0.0,
                    share=0.0, equal_dollars=None, risk_share=None, equal_risk_share=None,
                    current_capital_usd=current, min_capital_usd=None, reason="",
                    paper_ready=ready, floor=floor)

    p = Proposal(version="v", capital=3000.0, pool=0.0, held=0.0, min_days=28,
                 lines=[line("Bv2", "paper", bv2.to_dict(), ready=True),
                        line("W", "stand", weights_only.to_dict()),
                        line("FRAB", "live", None, current=90.0)],
                 correlations=[], correlation_note="", ladder={}, needs_decision=[])
    text = "\n".join(start_lines(p))
    assert "Bv2: работает от $150" in text and "все позиции не меньше минимального ордера" in text
    assert "— от $150" in text and "По правилу бумаги готова" in text
    assert "W: код не знает своей суммы" in text and "$640" in text
    assert "FRAB: минимум, при котором она работает, не измерен. Сейчас на настоящих деньгах $90" \
        in text
    assert working_minimum(bv2.to_dict(), 99.0) == 150.0 and working_minimum(None, 99.0) == 99.0
