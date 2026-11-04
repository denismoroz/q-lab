"""Doubles of an idea (`qlab.duplicates`, docs/DUPLICATES.md): the fingerprint of a spec before a
run, identical results after it. Fixtures: the same code and numbers in different wrappers,
defaults filled in, the same results under different numbers, results near but not equal."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import duplicates as dup
from qlab import forward_series
from qlab.ladder import Ledgers
from qlab.pipeline.spec import StrategySpec
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
)

FAKE = "fake.mod:Fake"
WRAP = "fake.mod:Wrap"
RETUNE = "qlab.strategies.retune:Retune"


class Fake:
    """A strategy that declares its own defaults."""

    DEFAULT_PARAMS = {"lookback": 30, "threshold": 2.0, "coins": ["BTC", "ETH"]}


@pytest.fixture(autouse=True)
def _known_strategies(monkeypatch) -> None:
    monkeypatch.setattr(dup, "_strategy_class", lambda code: Fake if code == FAKE else None)


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def spec(idea_id: str = "a", *, code: str = FAKE, params: dict | None = None, title: str = "t",
         source: str = "binance", interval: str = "1d", taker: float = 3.5, slippage: float = 0.9,
         money: float | None = None, choice: dict | None = None, **extra) -> StrategySpec:
    body = {
        "idea_id": idea_id, "title": title, "code_ref": code,
        "params": {"lookback": 30} if params is None else params,
        "data": {"source": source, "interval": interval, "start": "2020-01-01",
                 "end": "2026-10-01"},
        "costs": {"taker_fee_bps": taker, "slippage_bps": slippage},
        "min_leg_notional": 5.0, "unexpressed_mechanisms": [], **extra}
    if money is not None:
        body.update(strategy_capital_usd=money, sources={"strategy_capital_usd": "owner"})
    if choice is not None:
        body["choice"] = choice
    return StrategySpec.model_validate(body)


def retune_form(base: dict, grid: dict, *, code: str = FAKE, idea_id: str = "r") -> StrategySpec:
    return spec(idea_id, code=RETUNE, params={
        "inner_code_ref": code, "base_params": base, "grid": grid, "refit_every": "M",
        "window": "expanding", "objective": "sharpe_net",
        "score_costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9}})


# --------------------------------------------------------------------------
# The fingerprint


def test_ids_titles_and_texts_are_not_part_of_the_fingerprint() -> None:
    one = spec("first", title="one title")
    two = spec("second", title="quite another title", plain_title="plain words")
    assert dup.fingerprint(one).key() == dup.fingerprint(two).key()


def test_defaults_filled_in_read_the_same_as_defaults_left_out() -> None:
    left_out = spec(params={"lookback": 30})
    written = spec(params={"lookback": 30, "threshold": 2, "coins": ["BTC", "ETH"]})
    assert dup.fingerprint(left_out).key() == dup.fingerprint(written).key()
    other = spec(params={"lookback": 30, "threshold": 3.0})
    assert dup.fingerprint(left_out).key() != dup.fingerprint(other).key()
    assert "params" not in dup.fingerprint(left_out).same_parts(dup.fingerprint(other))


def test_a_choice_form_and_the_retune_form_of_the_same_lists_are_the_same_idea() -> None:
    declared = spec("declared", params={"lookback": 10}, choice={
        "axes": {"lookback": [10, 20]}, "source": "night/own_numbers.yaml"})
    stored = retune_form({"lookback": 10}, {"lookback": {"short": {"lookback": 10},
                                                          "long": {"lookback": 20}}})
    assert dup.fingerprint(declared).key() == dup.fingerprint(stored).key()
    # other labels, the lists in another order: still the same variants
    relabelled = retune_form({"lookback": 10}, {"lookback": {"b": {"lookback": 20},
                                                              "a": {"lookback": 10}}})
    assert dup.fingerprint(relabelled).key() == dup.fingerprint(stored).key()
    # one more value is another list to choose among
    wider = spec("wider", params={"lookback": 10}, choice={
        "axes": {"lookback": [10, 20, 40]}, "source": "night/own_numbers.yaml"})
    assert dup.fingerprint(wider).key() != dup.fingerprint(declared).key()
    # and a spec with no list is not an idea that chooses
    assert dup.fingerprint(spec(params={"lookback": 10})).key() != dup.fingerprint(declared).key()


def test_named_candidates_read_as_their_full_variants() -> None:
    named = spec("named", params={"lookback": 10}, choice={
        "candidates": {"x": {"lookback": 10}, "y": {"lookback": 20}}, "source": "card"})
    axes = spec("axes", params={"lookback": 10}, choice={
        "axes": {"lookback": [10, 20]}, "source": "card"})
    assert dup.fingerprint(named).key() == dup.fingerprint(axes).key()


def test_the_inner_strategy_is_taken_out_of_a_wrapper_but_what_the_wrapper_does_stays() -> None:
    plain = spec("plain", params={"lookback": 10})
    cadence = spec("cadence", code=WRAP, params={
        "inner_code_ref": FAKE, "inner_params": {"lookback": 10},
        "entry_every": "1D", "exit_every": "1h"})
    fp_plain, fp_cadence = dup.fingerprint(plain), dup.fingerprint(cadence)
    assert fp_cadence.code == fp_plain.code == FAKE  # the inner code is what is compared
    assert fp_cadence.parts["params"] == fp_plain.parts["params"]
    assert fp_cadence.key() != fp_plain.key()  # a cadence changes what is held
    assert "wrappers" not in fp_cadence.same_parts(fp_plain)
    same_cadence = spec("again", code=WRAP, params={
        "inner_code_ref": FAKE, "inner_params": {"lookback": 10, "threshold": 2.0},
        "exit_every": "1h", "entry_every": "1D"})
    assert dup.fingerprint(same_cadence).key() == fp_cadence.key()
    slower = spec("slower", code=WRAP, params={
        "inner_code_ref": FAKE, "inner_params": {"lookback": 10},
        "entry_every": "1W", "exit_every": "1h"})
    assert dup.fingerprint(slower).key() != fp_cadence.key()


def test_a_wrapper_inside_a_retune_is_unwrapped_for_every_variant() -> None:
    gated = lambda n: {"inner_code_ref": FAKE, "inner_params": {"lookback": n},  # noqa: E731
                       "entry_every": "1D", "exit_every": "1h"}
    stored = retune_form(gated(10), {"x": {"a": {"inner_params.lookback": 10},
                                           "b": {"inner_params.lookback": 20}}}, code=WRAP)
    declared = spec("declared", code=WRAP, params=gated(10), choice={
        "axes": {"inner_params.lookback": [10, 20]}, "source": "card"})
    assert dup.fingerprint(stored).key() == dup.fingerprint(declared).key()


@pytest.mark.parametrize(("change", "part"), [
    ({"source": "hyperliquid"}, "data"),
    ({"interval": "1h"}, "data"),
    ({"taker": 4.5}, "costs"),
    ({"slippage": 1.5}, "costs"),
    ({"money": 100.0}, "money"),
    ({"params": {"lookback": 60}}, "params"),
    ({"code": "fake.mod:Other"}, "code"),
])
def test_each_difference_is_in_its_own_part(change, part) -> None:
    base = dup.fingerprint(spec("a"))
    other = dup.fingerprint(spec("b", **change))
    assert base.key() != other.key()
    assert part not in base.same_parts(other)


def test_data_details_are_part_of_the_fingerprint() -> None:
    base = spec("a")

    def with_data(**data) -> StrategySpec:
        body = base.model_dump(mode="json", exclude_none=True)
        body.update(idea_id="c", data={**base.data.model_dump(mode="json"), **data})
        return StrategySpec.model_validate(body)

    key = dup.fingerprint(base).key()
    assert dup.fingerprint(with_data()).key() == key
    for change in ({"include_spot": True}, {"include_spot": True, "spot_prices": "own"},
                   {"start": "2021-01-01"}, {"instruments": ["BTC", "ETH"]},
                   {"min_daily_volume_usd": 1e6}):
        assert dup.fingerprint(with_data(**change)).key() != key, change
    # the same instruments in another order are the same list
    assert (dup.fingerprint(with_data(instruments=["BTC", "ETH"])).key()
            == dup.fingerprint(with_data(instruments=["ETH", "BTC"])).key())


def test_numbers_that_are_equal_read_the_same() -> None:
    assert (dup.fingerprint(spec("a", params={"lookback": 30})).key()
            == dup.fingerprint(spec("b", params={"lookback": 30.0})).key())


# --------------------------------------------------------------------------
# The open ideas and the refusal before a run


def add_idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.BENCH, *, run: bool = True,
             day: int = 1, params: dict | None = None) -> int | None:
    session.add(Idea(id=idea_id, title=f"title of {idea_id}", source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     created_at=datetime(2026, 10, day, tzinfo=UTC)))
    session.flush()
    if not run:
        return None
    row = repo.add_spec(session, idea_id=idea_id, version=1, params=params or {"lookback": 30},
                        data_requirements={"source": "binance", "interval": "1d",
                                           "start": "2020-01-01", "end": "2026-10-01"},
                        rebalance="1d", costs_model={"taker_fee_bps": 3.5, "slippage_bps": 0.9},
                        code_ref=FAKE)
    trial = repo.add_trial(session, spec_id=row.id, config_hash=f"h-{idea_id}", code_sha="x",
                           params={}, status=TrialStatus.OK, route=TrialRoute.PAPER,
                           route_reason="r", metrics={}, source=TrialSource.IMPORTED,
                           stand_version="x", started_at=datetime(2026, 10, day, tzinfo=UTC))
    return trial.id


def write_spec(root: Path, s: StrategySpec) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{s.idea_id}.yaml").write_text(
        yaml.safe_dump(s.model_dump(mode="json", exclude_none=True), sort_keys=False,
                       allow_unicode=True), encoding="utf-8")


def test_a_new_spec_equal_to_an_open_ideas_latest_spec_is_refused(session, tmp_path) -> None:
    add_idea(session, "old")
    write_spec(tmp_path, spec("old", title="the old one"))
    index = dup.open_index(session, spec_root=tmp_path)
    double = dup.refusal(spec("new", title="a new title", params={"lookback": 30, "threshold": 2}),
                         index)
    assert double is not None and double.of == "old"
    assert "дубль идеи old" in double.text() and "код fake.mod:Fake" in double.text()
    assert "код" in double.parts and "числа после умолчаний" in double.parts
    assert dup.refusal(spec("new", params={"lookback": 31}), index) is None
    assert dup.refusal(spec("old"), index) is None  # an idea is not its own double


def test_a_closed_idea_and_an_idea_never_run_are_not_compared_with(session, tmp_path) -> None:
    add_idea(session, "retired", IdeaStatus.RETIRED)
    add_idea(session, "unrun", run=False)
    for name in ("retired", "unrun"):
        write_spec(tmp_path, spec(name))
    assert dup.open_index(session, spec_root=tmp_path) == []
    assert dup.refusal(spec("new"), dup.open_index(session, spec_root=tmp_path)) is None


def test_an_idea_without_a_spec_file_is_read_from_its_latest_stored_row(session, tmp_path) -> None:
    add_idea(session, "stored", params={"lookback": 30})
    index = dup.open_index(session, spec_root=tmp_path)  # no spec files at all
    assert [i.idea_id for i in index] == ["stored"] and index[0].source == "registry"
    # the money and the smallest leg are not stored, so a spec that states them is not its double
    assert dup.refusal(spec("new", money=90.0), index) is None


def test_the_retune_row_of_an_open_idea_is_its_declared_lists(session, tmp_path) -> None:
    add_idea(session, "chooser", run=False)
    stored = retune_form({"lookback": 10}, {"lookback": {"s": {"lookback": 10},
                                                          "l": {"lookback": 20}}})
    row = repo.add_spec(session, idea_id="chooser", version=1, params=stored.params,
                        data_requirements={"source": "binance", "interval": "1d",
                                           "start": "2020-01-01", "end": "2026-10-01"},
                        rebalance="1d", costs_model={"taker_fee_bps": 3.5, "slippage_bps": 0.9},
                        code_ref=RETUNE)
    repo.add_trial(session, spec_id=row.id, config_hash="h", code_sha="x", params={},
                   status=TrialStatus.OK, route=TrialRoute.PAPER, route_reason="r", metrics={},
                   source=TrialSource.IMPORTED, stand_version="x",
                   started_at=datetime(2026, 10, 1, tzinfo=UTC))
    index = dup.open_index(session, spec_root=tmp_path)
    new = spec("new", params={"lookback": 10}, min_leg_notional=5.0, choice={
        "axes": {"lookback": [10, 20]}, "source": "night/own_numbers.yaml"})
    # the stored row has no smallest leg: only the file form can be compared in full
    assert dup.fingerprint(new, min_leg=-1.0).key() == index[0].fingerprint.key()


def test_a_known_idea_is_never_refused_and_a_new_one_is_found_by_its_trials(session) -> None:
    add_idea(session, "old")
    add_idea(session, "registered-not-run", run=False)
    assert dup.is_new(session, "registered-not-run") and dup.is_new(session, "never-seen")
    assert not dup.is_new(session, "old")
    assert dup.refusal_for_new_idea(spec("old"), session, index=[]) is None


def test_a_batch_compares_its_own_specs_with_each_other(session) -> None:
    index = dup.open_index(session, spec_root=Path("/nonexistent"))
    first = spec("first")
    assert dup.refusal(first, index) is None
    index.append(dup.as_open(first))
    assert dup.refusal(spec("second"), index).of == "first"  # type: ignore[union-attr]


def test_the_prompt_block_lists_open_ideas_in_plain_words() -> None:
    index = [dup.as_open(spec("a", title="Trend on caps", plain_title="Следует за ростом"))]
    text = dup.prompt_block(index)
    assert "- a: Следует за ростом" in text and "refused" in text


def test_the_refusal_is_remembered_for_the_report(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    double = dup.Duplicate("old", "t", ["код"], "words")
    dup.note_refusal(spec("new"), double, "researcher x", path)
    assert dup.load_ledger(path)["refused"]["new"]["of"] == "old"


# --------------------------------------------------------------------------
# After a run: identical results


def curve_payload(trial_id: int, daily: np.ndarray, *, venue: str = "binance",
                  start: str = "2025-01-01", reinvested: bool = True) -> dict:
    days = pd.date_range(start, periods=len(daily), freq="D")
    cum = (np.cumprod(1.0 + daily) - 1.0) if reinvested else np.cumsum(daily)
    points = [[(days[0] - timedelta(days=1)).date().isoformat(), 0.0, None]]
    points += [[d.date().isoformat(), float(c), None] for d, c in zip(days, cum, strict=True)]
    return {"format": forward_series.FORMAT, "trial_id": trial_id, "venue": venue,
            "reinvested": reinvested, "points": points}


def series(seed: int = 1, days: int = 200) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.001, 0.01, days)


def test_daily_returns_are_read_back_from_the_stored_curve() -> None:
    x = series()
    for reinvested in (True, False):
        back = dup.daily_returns(curve_payload(1, x, reinvested=reinvested))
        np.testing.assert_allclose(back.to_numpy(), x, atol=1e-12)


def test_the_same_daily_money_is_a_match() -> None:
    x = series()
    same = dup.compare_series(curve_payload(1, x), curve_payload(2, x.copy()))
    assert same is not None and same.correlation > 0.9999 and same.annual_diff < 1e-9


def test_results_near_but_not_equal_are_not_flagged() -> None:
    x = series()
    base = curve_payload(1, x)
    noisy = x + np.random.default_rng(7).normal(0, 0.06 * x.std(), len(x))
    assert 0.99 < np.corrcoef(x, noisy)[0, 1] < 0.999
    assert dup.compare_series(base, curve_payload(2, noisy)) is None  # correlation too low
    shifted = x + 0.0002  # the same shape, a yearly return that differs by points
    assert dup.compare_series(base, curve_payload(3, shifted)) is None
    assert dup.compare_series(base, curve_payload(4, series(seed=2))) is None  # another book
    tiny = x + 1e-6  # 0.04 pp a year: inside the tolerance, so a match
    assert dup.compare_series(base, curve_payload(5, tiny)) is not None


def test_another_venue_another_window_or_a_flat_book_is_not_a_match() -> None:
    x = series()
    base = curve_payload(1, x)
    assert dup.compare_series(base, curve_payload(2, x, venue="hyperliquid")) is None
    assert dup.compare_series(base, curve_payload(3, x[:20])) is None  # too few common days
    assert dup.compare_series(base, curve_payload(4, x, start="2025-06-01")) is None  # window
    flat = curve_payload(5, np.zeros(200))
    assert dup.compare_series(flat, curve_payload(6, np.zeros(200))) is None


def test_a_new_run_with_the_same_money_under_other_numbers_is_marked(session, tmp_path) -> None:
    store, ledger = tmp_path / "series", tmp_path / "ledger.json"
    x = series()
    old = add_idea(session, "old", day=1, params={"lookback": 30})
    new = add_idea(session, "new", day=2, params={"lookback": 99})  # other numbers, same money
    other = add_idea(session, "other", day=3, params={"lookback": 5})
    forward_series.save(curve_payload(old, x), store)
    forward_series.save(curve_payload(new, x.copy()), store)
    forward_series.save(curve_payload(other, series(seed=5)), store)

    record = dup.mark_by_results(session, "new", new, store=store, path=ledger)
    assert record is not None and record["of"] == "old" and record["of_trial_id"] == old
    assert dup.load_ledger(ledger)["results"]["new"]["of"] == "old"
    assert dup.mark_by_results(session, "other", other, store=store, path=ledger) is None
    assert "other" not in dup.load_ledger(ledger)["results"]
    # the older idea is never marked against the newer one
    assert dup.mark_by_results(session, "old", old, store=store, path=ledger) is None
    # every check is remembered, a double or not: «not a double» differs from «not checked yet»
    checked = dup.load_ledger(ledger)["checked"]
    assert set(checked) == {"new", "other", "old"} and checked["new"]["trial_id"] == new

    # a later run of "new" that no longer matches takes the mark off
    again = add_idea_run(session, "new", 9)
    forward_series.save(curve_payload(again, series(seed=9)), store)
    assert dup.mark_by_results(session, "new", again, store=store, path=ledger) is None
    assert "new" not in dup.load_ledger(ledger)["results"]


def add_idea_run(session, idea_id: str, version: int) -> int:
    row = repo.add_spec(session, idea_id=idea_id, version=version, params={"lookback": version},
                        data_requirements={"source": "binance"}, rebalance="1d", costs_model={},
                        code_ref=FAKE)
    return repo.add_trial(session, spec_id=row.id, config_hash=f"h{version}", code_sha="x",
                          params={}, status=TrialStatus.OK, route=TrialRoute.PAPER,
                          route_reason="r", metrics={}, source=TrialSource.IMPORTED,
                          stand_version="x", started_at=datetime(2026, 10, 9, tzinfo=UTC)).id


def test_a_closed_idea_is_not_compared_with(session, tmp_path) -> None:
    store = tmp_path / "series"
    x = series()
    old = add_idea(session, "old", IdeaStatus.RETIRED, day=1)
    new = add_idea(session, "new", day=2)
    forward_series.save(curve_payload(old, x), store)
    forward_series.save(curve_payload(new, x), store)
    assert dup.find_same_results(session, "new", new, store=store) is None


def test_the_morning_report_proposes_only_doubles_whose_original_is_open(session, tmp_path) -> None:
    add_idea(session, "old", day=1)
    add_idea(session, "new", day=2)
    add_idea(session, "gone", IdeaStatus.RETIRED, day=3)
    record = {"of": "old", "trial_id": 2, "of_trial_id": 1, "venue": "binance",
              "correlation": 0.99999, "annual_diff_pp": 0.002, "days": 200, "annual": 0.1}
    ledger = {"results": {"new": record, "gone": record}, "refused": {}}
    lines = dup.report_lines(session, ledger)
    text = "\n".join(lines)
    assert "`new`" in text and "дубль old по результатам" in text and "`gone`" not in text
    assert "qlab retire new" in text and "не решение" in text
    ledger["results"]["new"] = {**record, "of": "gone"}  # the original is closed: nothing to say
    assert "`new`" not in "\n".join(dup.report_lines(session, ledger))


def test_the_ladder_shows_the_mark_in_step_five(session) -> None:
    from qlab import ladder as lad
    from qlab.test_ladder import idea as ladder_idea
    from qlab.test_ladder import run as ladder_run

    ladder_idea(session, "new", IdeaStatus.BENCH)
    ladder_run(session, "new", TrialRoute.PAPER)
    record = {"of": "old", "trial_id": 2, "of_trial_id": 1, "venue": "binance",
              "correlation": 0.99999, "annual_diff_pp": 0.002, "days": 200, "annual": 0.1}
    found = lad.build_ladder(session, "new", Ledgers(duplicates={"new": record}))
    names = [r.name for r in found.steps[4].readings]
    assert "duplicate" in names
    reading = next(r for r in found.steps[4].readings if r.name == "duplicate")
    assert reading.recorded and "дубль old по результатам" in reading.text
    clean = lad.build_ladder(session, "new", Ledgers())
    assert "duplicate" not in [r.name for r in clean.steps[4].readings]


def test_the_registry_groups(session, tmp_path) -> None:
    store = tmp_path / "series"
    x = series()
    a = add_idea(session, "a", day=1, params={"lookback": 30})
    b = add_idea(session, "b", day=2, params={"lookback": 30})
    c = add_idea(session, "c", day=3, params={"lookback": 40})
    d = add_idea(session, "d", day=4, params={"lookback": 50})
    for trial, daily in ((a, x), (b, x), (c, x), (d, series(seed=3))):
        forward_series.save(curve_payload(trial, daily), store)
    index = dup.open_index(session, spec_root=tmp_path / "none")
    by_print = dup.groups_by_fingerprint(index)
    assert [g.ideas for g in by_print] == [["a", "b"]]
    by_results = dup.groups_by_results(session, index, store=store)
    assert [g.ideas for g in by_results] == [["a", "b", "c"]]  # c has other numbers, same money
    json.dumps([g.detail for g in by_results])  # printable
