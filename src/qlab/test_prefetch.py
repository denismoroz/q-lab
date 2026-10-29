"""Tests for `qlab.prefetch`: the watch list's data is fetched in a background thread while earlier
items compute (docs/NIGHT_SPEED.md), without fetching anything twice."""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

import yaml

from qlab import night, prefetch
from qlab.pipeline.evaluate import _request_lock
from qlab.pipeline.spec import StrategySpec, load_spec

SPEC_DIR = Path(__file__).resolve().parents[2] / "specs"


def _spec(idea: str, end: date = date(2026, 1, 1)) -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": idea, "title": "t", "code_ref": "x.y:Z", "params": {},
        "data": {"source": "hyperliquid", "interval": "1d", "start": date(2025, 1, 1),
                 "end": end, "instruments": None},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 12.0,
        "unexpressed_mechanisms": [], "sources": {"p": "test"}})


def test_specs_are_fetched_in_order_and_the_report_says_which(monkeypatch) -> None:
    seen: list[str] = []
    fetcher = prefetch.Prefetcher([_spec("a"), _spec("b"), _spec("c")],
                                  fetch=lambda s: seen.append(s.idea_id)).start()
    fetcher.join(10)
    report = fetcher.report()
    assert seen == ["a", "b", "c"] and report["done"] == ["a", "b", "c"]
    assert report["failed"] == {} and report["planned"] == 3 and report["finished_at"]


def test_a_failed_fetch_is_noted_and_the_next_one_still_runs() -> None:
    def fetch(spec):
        if spec.idea_id == "b":
            raise RuntimeError("venue down")

    fetcher = prefetch.Prefetcher([_spec("a"), _spec("b"), _spec("c")], fetch=fetch).start()
    fetcher.join(10)
    report = fetcher.report()
    assert report["done"] == ["a", "c"]
    assert report["failed"] == {"b": "RuntimeError: venue down"}


def test_stop_finishes_the_fetch_in_hand_and_starts_no_other() -> None:
    inside, release = threading.Event(), threading.Event()
    seen: list[str] = []

    def fetch(spec):
        seen.append(spec.idea_id)
        inside.set()
        release.wait(10)

    fetcher = prefetch.Prefetcher([_spec("a"), _spec("b")], fetch=fetch).start()
    assert inside.wait(10)
    stopper = threading.Thread(target=fetcher.stop, kwargs={"timeout": 10})
    stopper.start()
    release.set()
    stopper.join(10)
    assert seen == ["a"] and not fetcher.alive


def test_it_is_on_unless_the_environment_says_off(monkeypatch) -> None:
    monkeypatch.delenv(prefetch.PREFETCH_ENV, raising=False)
    assert prefetch.enabled()
    for off in ("0", "false", "OFF", "no"):
        monkeypatch.setenv(prefetch.PREFETCH_ENV, off)
        assert not prefetch.enabled()
    assert prefetch.start([_spec("a")]) is None  # off: nothing starts
    monkeypatch.delenv(prefetch.PREFETCH_ENV)
    assert prefetch.start([]) is None  # on, but nothing to fetch


def test_one_data_request_has_one_lock_and_another_request_another() -> None:
    a = _spec("a").data
    # The data decides, not the idea.
    assert _request_lock(a) is _request_lock(_spec("other idea").data)
    assert _request_lock(a) is not _request_lock(_spec("a", end=date(2026, 1, 2)).data)


def test_the_night_prefetches_the_items_it_has_yet_to_run_with_the_nights_end(
        tmp_path, monkeypatch) -> None:
    watch = tmp_path / "watchlist.yaml"
    done_spec, todo_spec = SPEC_DIR / "trend.yaml", SPEC_DIR / "bv2.yaml"
    watch.write_text(yaml.safe_dump({"specs": [str(done_spec), str(todo_spec),
                                               str(tmp_path / "missing.yaml")]}))
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setenv(prefetch.PREFETCH_ENV, "1")
    started: list = []
    monkeypatch.setattr(prefetch, "start", lambda specs: started.append(list(specs)) or "handle")
    end = date(2030, 1, 1)
    handle = night.start_prefetch(end, {"done": {f"watch:{done_spec}": "x"}})
    assert handle == "handle"
    [specs] = started
    # The one not done, with the night's end; the one already done and the broken one left out.
    assert [s.idea_id for s in specs] == [load_spec(todo_spec).idea_id]
    assert specs[0].data.end == end


def test_the_night_without_prefetch_starts_nothing(monkeypatch) -> None:
    monkeypatch.setenv(prefetch.PREFETCH_ENV, "0")
    assert night.start_prefetch(date(2030, 1, 1), {"done": {}}) is None
    state: dict = {}
    night.stop_prefetch(None, state)
    assert state == {}


def test_stopping_keeps_the_report_in_the_nights_state() -> None:
    fetcher = prefetch.Prefetcher([_spec("a")], fetch=lambda s: None).start()
    state: dict = {}
    night.stop_prefetch(fetcher, state)
    assert state["prefetch"]["done"] == ["a"]


def test_two_asks_for_one_request_take_turns_and_other_requests_do_not_wait(monkeypatch) -> None:
    """The item and the prefetch thread asking for the same data: the second waits for the
    first's build and then finds the snapshot, so the data is fetched once."""
    import qlab.pipeline.evaluate as evaluate

    inside, release = threading.Event(), threading.Event()
    order: list[str] = []

    def slow(session, spec):
        order.append(f"start {spec.idea_id}")
        if spec.idea_id == "one":
            inside.set()
            release.wait(10)
        order.append(f"end {spec.idea_id}")
        return spec.idea_id

    monkeypatch.setattr(evaluate, "_resolve_panel", slow)
    first = threading.Thread(target=evaluate.resolve_panel, args=(None, _spec("one")))
    first.start()
    assert inside.wait(10)
    second = threading.Thread(target=evaluate.resolve_panel, args=(None, _spec("two")))
    second.start()
    # Another request (a different end) is not held up by the first.
    other = evaluate.resolve_panel(None, _spec("other", end=date(2026, 3, 3)))
    assert other == "other"
    assert order == ["start one", "start other", "end other"]
    release.set()
    first.join(10)
    second.join(10)
    assert order[3:] == ["end one", "start two", "end two"]


def test_one_fetch_at_a_time_from_a_venue_but_venues_do_not_wait_for_each_other() -> None:
    from qlab.data.snapshot import _venue_lock

    assert _venue_lock("hyperliquid") is _venue_lock("hyperliquid")
    assert _venue_lock("hyperliquid") is not _venue_lock("binance")
