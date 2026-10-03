"""Tests for `qlab.night`: dates, data extension, resume, report."""

from __future__ import annotations

from datetime import UTC, date, datetime

import qlab.night as night
from qlab.night import ItemResult, last_closed_day, with_end
from qlab.pipeline.spec import StrategySpec


def _spec() -> StrategySpec:
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": {},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-09-20"},
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def test_last_closed_day_and_extension_never_shortens() -> None:
    assert last_closed_day(datetime(2026, 10, 3, 5, 30, tzinfo=UTC)) == date(2026, 10, 2)
    assert with_end(_spec(), date(2026, 10, 2)).data.end == date(2026, 10, 2)
    assert with_end(_spec(), date(2026, 9, 1)).data.end == date(2026, 9, 20)


def test_a_stopped_night_resumes_where_it_left_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: [a.yaml, b.yaml]\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "load_spec",
                        lambda p: _spec().model_copy(update={"idea_id": str(p)}))
    calls = []

    def fake_eval(kind, spec, **kw):
        calls.append(spec.idea_id)
        if spec.idea_id == "b.yaml" and len(calls) == 2:
            raise KeyboardInterrupt  # the night is stopped mid-run
        return ItemResult(kind=kind, name=spec.idea_id, route="needs-forward",
                          metrics={"ann_return_net": 0.05, "sharpe_net": 1.0})

    monkeypatch.setattr(night, "evaluate_item", fake_eval)
    day = date(2026, 10, 3)
    try:
        night.run(day, graveyard=False, paper=False)
    except KeyboardInterrupt:
        pass
    report = night.run(day, graveyard=False, paper=False)
    assert calls == ["a.yaml", "b.yaml", "b.yaml"]  # a was not redone
    text = report.read_text()
    assert "a.yaml" in text and "b.yaml" in text and "Токены" in text
