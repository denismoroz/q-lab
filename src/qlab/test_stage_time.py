"""The time limit of the night's stages that run ideas on the stand from a list
(`qlab.stage_time`, docs/PIPELINE.md, «Лимит времени на шаг»): the clock, the ledger of how long
each item took, the report's words, and the stages that use them (waiting, twins, overlays and the
watch list)."""

from __future__ import annotations

import types

import pytest

import qlab.registry.db as db_module
from qlab import night, stage_time, waiting
from qlab.registry.models import Base
from qlab.stage_time import SecondsLedger, StageClock, page_note, read_limit, time_lines
from qlab.test_waiting import DAY, END, _world


class FakeTime:
    """Seconds moved by hand."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def minutes(self, n: float) -> None:
        self.t += n * 60


def _clock(limit=45.0, **kw):
    fake = FakeTime()
    return StageClock("waiting", limit, now=fake, hand_command="qlab twins run --only {idea}",
                      **kw), fake


# ---- the clock -------------------------------------------------------------------------


def test_a_stage_without_a_limit_refuses_nothing() -> None:
    clock, fake = _clock(None)
    fake.minutes(500)
    assert clock.refuse("any", 10_000.0) is None and clock.summary()["limit_minutes"] is None


def test_an_item_larger_than_the_whole_limit_is_heavy_and_named_with_the_hand_command() -> None:
    clock, _ = _clock(45.0)
    kind, text = clock.refuse("bv2-top10:stage2", 72.0, "в прошлый раз 72 минуты")
    assert kind == stage_time.HEAVY
    assert "72 минут" in text and "45" in text and "qlab twins run --only bv2-top10:stage2" in text
    assert clock.summary()["heavy"] == {"bv2-top10:stage2": text}


def test_an_item_larger_than_what_is_left_waits_for_the_next_night() -> None:
    clock, fake = _clock(45.0)
    fake.minutes(30)
    kind, text = clock.refuse("mid", 20.0)
    assert kind == stage_time.DEFERRED and "15 из 45" in text
    assert clock.refuse("small", 10.0) is None  # fits in what is left
    assert clock.summary()["deferred"] == ["mid"] and "heavy" not in clock.summary()


def test_once_the_time_is_up_nothing_new_starts() -> None:
    clock, fake = _clock(45.0)
    fake.minutes(46)
    kind, _ = clock.refuse("x")
    assert kind == stage_time.TIME_UP and clock.summary()["time_up"] is True


def test_the_estimate_defaults_to_the_time_the_item_took_last_time(tmp_path) -> None:
    ledger = SecondsLedger(tmp_path / "s.json")
    ledger.record("waiting", "slow", 3600, "2026-11-01")
    clock, _ = _clock(45.0, ledger=ledger)
    assert clock.refuse("slow")[0] == stage_time.HEAVY
    assert clock.refuse("never-run") is None


def test_an_item_that_takes_the_stage_past_its_limit_is_recorded_not_cut(tmp_path) -> None:
    ledger = SecondsLedger(tmp_path / "s.json")
    clock, fake = _clock(45.0, ledger=ledger, day="2026-11-01")
    assert clock.refuse("long") is None  # nothing known about it: it starts
    fake.minutes(50)
    clock.finished("long", 50 * 60)
    summary = clock.summary()
    assert summary["overran"] == {"idea": "long", "minutes": 50.0, "limit_minutes": 45.0}
    # the next night knows how long it took and does not start it again
    again, _ = _clock(45.0, ledger=SecondsLedger(tmp_path / "s.json"))
    assert again.refuse("long")[0] == stage_time.HEAVY


def test_a_fast_run_clears_the_ledger_of_a_slow_past(tmp_path) -> None:
    ledger = SecondsLedger(tmp_path / "s.json")
    ledger.record("waiting", "item", 4000, "2026-10-31")
    ledger.record("waiting", "item", 120, "2026-11-01")  # a hand run after the fix
    assert ledger.last("waiting", "item") == 120.0
    clock, _ = _clock(45.0, ledger=ledger)
    assert clock.refuse("item") is None


def test_the_watch_list_is_timed_but_never_refused() -> None:
    fake = FakeTime()
    clock = StageClock("recheck", 60.0, slow_minutes=15.0, now=fake)
    for name, minutes in (("a", 5.0), ("b", 17.0), ("c", 40.0)):
        fake.minutes(minutes)
        clock.finished(name, minutes * 60)
    summary = clock.summary()
    assert summary["slow"] == [{"idea": "b", "minutes": 17.0}, {"idea": "c", "minutes": 40.0}]
    assert summary["overran"]["idea"] == "c" and summary["minutes_spent"] == 62.0
    lines = time_lines(summary, watch=True)
    assert any("62 минут" in line and "не сокращают" in line for line in lines)
    assert any("b: шла 17 минут" in line for line in lines)
    assert page_note(summary, watch=True)


def test_the_limit_is_read_from_the_stages_own_config(tmp_path) -> None:
    path = tmp_path / "w.yaml"
    assert read_limit(path) is None  # no file: no limit
    path.write_text("enabled: true\n", encoding="utf-8")
    assert read_limit(path) is None
    path.write_text("max_minutes_per_night: 45\nslow_item_minutes: 15\n", encoding="utf-8")
    assert read_limit(path) == 45.0 and read_limit(path, stage_time.SLOW_KEY) == 15.0
    path.write_text("max_minutes_per_night: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        read_limit(path)


def test_every_stage_that_walks_a_list_has_its_limit_in_its_config() -> None:
    from pathlib import Path

    for name in ("waiting", "twins", "overlays", "watchlist", "restamp", "second_venue"):
        assert read_limit(Path("night") / f"{name}.yaml") is not None, name
    assert read_limit(Path("night/watchlist.yaml"), stage_time.SLOW_KEY) is not None


# ---- waiting ----------------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    Base.metadata.create_all(db_module.get_engine())
    yield db_module
    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture()
def timed(tmp_path, monkeypatch):
    """The waiting stage with a clock moved by hand: each stand run takes `cost[idea]` minutes."""
    from qlab.night import ItemResult
    from qlab.pipeline import stage2

    fake = FakeTime()
    cost: dict[str, float] = {}
    calls: list[str] = []

    def fake_eval(kind, spec, *, end, noise_trials):
        calls.append(spec.idea_id)
        fake.minutes(cost.get(spec.idea_id, 1.0))
        return ItemResult(kind=kind, name=spec.idea_id, trial_id=900, route="needs-forward",
                          previous_route="needs-forward",
                          metrics={"forward_days": 100.0, "forward_days_needed": 100.0})

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    monkeypatch.setattr(stage2, "load_config", lambda: None)
    monkeypatch.setattr("qlab.rules.loader.load_latest",
                        lambda: types.SimpleNamespace(regime_coverage_days=30, version="r1"))
    config = tmp_path / "waiting.yaml"
    config.write_text("enabled: true\ndecided_recheck_days: 7\ndecided_max_per_night: 4\n"
                      "max_minutes_per_night: 45\n", encoding="utf-8")
    real = waiting._stage_clock

    def clock(day, only, force, config_path):
        made = real(day, only, force, config_path)
        made._now = fake
        made._started = fake()
        return made

    monkeypatch.setattr(waiting, "_stage_clock", clock)
    return types.SimpleNamespace(fake=fake, cost=cost, calls=calls, config=config,
                                 tmp=tmp_path)


def _run_waiting(w, state=None, **kw):
    state = state if state is not None else {"done": {}}
    waiting.waiting_stage(DAY, END, state, noise_trials=1, spec_dir=w.tmp / "specs", watch=[],
                          config_path=w.config, **kw)
    return state


def test_an_idea_that_took_more_than_the_whole_limit_last_time_is_not_run_by_the_night(
        timed, registry) -> None:
    _world(timed.tmp, registry)
    SecondsLedger().record("waiting", "ripe:idea", 72 * 60, "2026-10-31")
    state = _run_waiting(timed)
    assert "ripe" not in timed.calls and "no-metric" in timed.calls  # the others still run
    row = next(e for e in state["waiting"] if e["idea"] == "ripe")
    assert row["action"] == "heavy" and "qlab twins run --only ripe" in row["why"]
    assert "ripe:idea" in state["waiting_time"]["heavy"]
    assert "ripe" in waiting.followed_ideas(state["waiting"])  # not silently dropped
    text = "\n".join(waiting.report_lines(state["waiting"], None, state["waiting_time"]))
    assert "слишком тяжёлых для ночи: 1" in text and "не запущена" in text
    assert "ripe" not in {k.split(":")[1] for k in state["done"]}  # not marked as done


def test_a_hand_run_of_the_heavy_idea_has_no_limit_and_shows_the_next_night_it_is_fast(
        timed, registry) -> None:
    _world(timed.tmp, registry)
    SecondsLedger().record("waiting", "ripe:idea", 72 * 60, "2026-10-31")
    state = _run_waiting(timed, only={"ripe"}, force=True)
    assert timed.calls == ["ripe"] and "waiting_time" not in state
    assert SecondsLedger().last("waiting", "ripe:idea") == 60.0  # one fake minute
    timed.calls.clear()
    _run_waiting(timed, {"done": {}})
    assert "ripe" in timed.calls  # the night takes it again


def test_nothing_new_starts_once_the_time_is_up_and_the_overrun_is_recorded(
        timed, registry) -> None:
    _world(timed.tmp, registry)
    timed.cost["no-metric"] = 50.0  # first in the plan's order: takes the whole limit
    state = _run_waiting(timed)
    assert timed.calls == ["no-metric"]
    time = state["waiting_time"]
    assert time["overran"]["idea"] == "no-metric:idea" and time["time_up"] is True
    assert "ripe:idea" in time["deferred"] and "no-metric:idea" not in time["deferred"]
    waits = {e["idea"]: e for e in state["waiting"]}
    assert waits["ripe"]["action"] == "wait" and "вышло" in waits["ripe"]["why"]
    assert "ripe" not in {k.split(":")[1] for k in state["done"]}
    note = stage_time.page_note(time, waiting.HAND_COMMAND)
    assert "no-metric" in note and "qlab twins run --only no-metric" in note
