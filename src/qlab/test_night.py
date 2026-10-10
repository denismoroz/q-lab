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
        night.run(day, graveyard=False, paper=False, implement=False, search=False,
                  detectors=False)
    except KeyboardInterrupt:
        pass
    report = night.run(day, graveyard=False, paper=False, implement=False, search=False,
                       detectors=False)
    assert calls == ["a.yaml", "b.yaml", "b.yaml"]  # a was not redone
    text = report.read_text()
    assert "a.yaml" in text and "b.yaml" in text and "Токены" in text


def test_report_keeps_the_forward_test_out_of_yearly_figures() -> None:
    from qlab.night import _report_row

    # Nineteen forward days at +390% a year must not be shown as +390%.
    row = _report_row({"ann_return_net": 3.9, "sharpe_net": 6.4, "forward_days": 19.0,
                       "fit_ann_return_net": 0.068, "fit_sharpe_net": 0.97,
                       "fit_regime_bull_return": 0.31, "regime_flat_return": 0.066})
    yearly, sharpe, days, forward_total, bull, flat, bear = row
    assert (yearly, sharpe, days) == ("+6.8%", "0.97", "19")
    assert forward_total == f"{(1 + 3.9) ** (19 / 365.25) - 1:+.1%}"
    assert (bull, flat, bear) == ("+31.0%", "—", "—")  # the selection period's split only


def test_report_without_a_forward_test_shows_the_whole_window() -> None:
    from qlab.night import _report_row

    row = _report_row({"ann_return_net": 0.076, "sharpe_net": 1.07, "forward_days": 0.0,
                       "regime_bear_return": 0.124})
    assert row == ("+7.6%", "1.07", "0", "—", "—", "—", "+12.4%")


def test_detectors_are_rebuilt_before_the_watch_list_and_reported(tmp_path, monkeypatch) -> None:
    import qlab.detector_builds as db
    from qlab import night

    order = []
    config = tmp_path / "detectors.yaml"
    config.write_text("regimes: false\n")
    monkeypatch.setattr(db, "CONFIG", config)
    monkeypatch.setattr(db, "build_all", lambda end, config=config: order.append("detectors")
                        or [{"name": "level1 x", "ok": True, "note": "2020-01-01..2026-10-03"}])
    monkeypatch.setattr(night, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(night, "REPORT_DIR", tmp_path / "reports")
    watch = tmp_path / "watch.yaml"
    watch.write_text("specs: []\n")
    monkeypatch.setattr(night, "WATCHLIST", watch)
    monkeypatch.setattr(night, "_token_lines", lambda day: "")
    path = night.run(date(2026, 10, 4), graveyard=False, paper=False, implement=False,
                     search=False)
    assert order == ["detectors"]
    assert "level1 x" in path.read_text(encoding="utf-8")


def test_report_with_only_a_forward_test_and_with_a_nan() -> None:
    from qlab.night import _report_row

    row = _report_row({"ann_return_net": 0.1, "sharpe_net": 1.0, "forward_days": 365.25})
    assert row[:4] == ("—", "—", "365", "+10.0%")
    row = _report_row({"ann_return_net": float("nan"), "fit_ann_return_net": 0.05,
                       "fit_sharpe_net": 0.5, "forward_days": 2.0})
    assert row[3] == "—"


def test_bench_ideas_nobody_follows_are_named(tmp_path, monkeypatch) -> None:
    """2026-10-10: seventeen bench ideas were finished experiments whose
    forward tests could never grow. The report says which ideas wait in vain."""
    import qlab.registry.db as db_module
    from qlab import night
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Base, IdeaStatus, Profile, SourceType

    monkeypatch.setenv("QLAB_DB", str(tmp_path / "q.db"))
    db_module._engine = None
    db_module._SessionLocal = None
    Base.metadata.create_all(db_module.get_engine())
    with session_scope() as session:
        for idea_id in ("followed", "forgotten"):
            repo.upsert_idea(session, id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                             asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER)
            repo.set_status(session, idea_id=idea_id, new_status=IdeaStatus.BENCH)

    class _Spec:
        idea_id = "followed"

    monkeypatch.setattr(night, "load_spec", lambda path: _Spec())
    assert night.unwatched_bench(["specs/followed.yaml"]) == ["forgotten"]
    db_module._engine = None
    db_module._SessionLocal = None
