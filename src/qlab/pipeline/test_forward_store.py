"""The forward test's daily series is stored by the run that decides the verdict, and can be
rebuilt from the stored spec and snapshot (docs/FORWARD_CHART.md). Reuses the toy strategy,
fixture panel and registry of `test_evaluate`."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from qlab import forward_series
from qlab.pipeline import test_evaluate as base
from qlab.pipeline.evaluate import evaluate_spec
from qlab.registry.models import Trial

# The registry, panel and idea of `test_evaluate`, as fixtures of this module.
engine, session = base.engine, base.session
_idea_row, _no_network = base._idea_row, base._no_network
CAPITAL_FIT_GENEROUS, CAPITAL_FIT_TIGHT = base.CAPITAL_FIT_GENEROUS, base.CAPITAL_FIT_TIGHT
_explicit_long, _split_ruleset = base._explicit_long, base._split_ruleset


def _causal_spec(session, tmp_path):
    return _explicit_long(session, tmp_path, params_fixed_at=date(2026, 1, 3),
                          params_fixed_evidence="test", selects_causally=True)


RULESET = _split_ruleset([CAPITAL_FIT_GENEROUS, base.HONEST_UNIVERSE_INFO])


def test_a_run_that_passes_stores_the_forward_series_from_its_own_backtest(session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=RULESET)

    assert result.routing.route == "paper"
    stored = forward_series.load(result.trial_id)
    assert stored is not None and stored["built_by"] == "evaluate"
    assert stored["idea_id"] == spec.idea_id and stored["venue"] == "hyperliquid"
    # The same number the rules saw: the chart ends where the verdict's metrics do.
    assert stored["total_return"] == pytest.approx(result.metrics["total_return_net"])
    assert stored["matches_trial"] is True
    assert stored["points"][0][1] == 0.0 and len(stored["points"]) > 2
    assert stored["start"] == "2026-01-03"
    assert stored["judged_on_forward"] is True


def test_the_series_covers_the_whole_window_with_the_warmup_and_the_forward_part(
        session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=RULESET)

    stored = forward_series.load(result.trial_id)
    whole = stored["whole"]
    parts = {p["part"]: p for p in whole["parts"]}
    assert whole["start"] < stored["start"] and whole["end"] == stored["end"]
    assert "warmup" in parts and parts["forward"]["start"] == stored["start"] == whole["split_date"]
    # the forward part of the whole-window series is the part the rules read
    assert parts["forward"]["days"] == stored["days"]
    assert whole["points"][0][1] == 0.0 and len(whole["points"]) > len(stored["points"])


def test_a_refusal_stores_its_series_too(session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=_split_ruleset([CAPITAL_FIT_TIGHT]))

    assert result.routing.route == "reject"
    stored = forward_series.load(result.trial_id)
    assert stored is not None and forward_series.has_whole(stored)


def test_calibration_noise_stores_nothing(session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=RULESET,
                           update_idea_status=False)

    assert forward_series.load(result.trial_id) is None


def test_a_run_judged_on_its_fit_period_stores_the_window_as_selection_only(session, tmp_path):
    spec = _explicit_long(session, tmp_path)  # no fixed date: the whole window is the fit period

    result = evaluate_spec(spec, session=session, ruleset=RULESET)

    assert result.metrics.get("judged_on_forward") in (None, 0.0)
    stored = forward_series.load(result.trial_id)
    assert stored["judged_on_forward"] is False
    assert [p["part"] for p in stored["whole"]["parts"]] == ["selection"]
    assert stored["whole"]["split_date"] is None


def test_a_failure_to_write_the_series_does_not_fail_the_run(session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)

    def _boom(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(forward_series, "store_from_run", _boom)

    result = evaluate_spec(spec, session=session, ruleset=RULESET)

    assert result.routing.route == "paper" and result.error is None
    assert session.get(Trial, result.trial_id) is not None


def _write_spec(spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{spec.idea_id}.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")


def test_a_stored_run_is_replayed_from_its_spec_and_snapshot_to_the_same_series(
        session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    live = forward_series.load(result.trial_id)
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is not None, replayed.reason
    assert replayed.payload["built_by"] == "rebuild" and replayed.payload["matches_trial"] is True
    assert replayed.payload["points"] == live["points"]
    assert replayed.payload["whole"]["points"] == live["whole"]["points"]
    assert replayed.payload["whole"]["parts"] == live["whole"]["parts"]


def test_rebuild_writes_the_series_a_run_made_before_this_existed_lacks(
        session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    forward_series.series_path(result.trial_id).unlink()  # a run from before the series was kept
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    done = forward_series.rebuild(session, [spec.idea_id], spec_root=tmp_path / "specs")

    assert done == [(spec.idea_id, result.trial_id, "построен")]
    assert forward_series.load(result.trial_id)["built_by"] == "rebuild"


def test_rebuild_all_builds_a_refused_idea_and_redoes_a_forward_only_file(
        session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=_split_ruleset([CAPITAL_FIT_TIGHT]))
    assert result.routing.route == "reject"
    old = forward_series.load(result.trial_id)
    old["whole"], old["format"] = None, 1  # a file from before the whole window was kept
    forward_series.save(old)
    tight = _split_ruleset([CAPITAL_FIT_TIGHT])
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: tight)
    _write_spec(spec, tmp_path / "specs")
    seen = []

    done = forward_series.rebuild(session, every=True, spec_root=tmp_path / "specs",
                                  on_done=lambda *a: seen.append(a))

    assert (spec.idea_id, result.trial_id, "построен") in done and seen
    assert forward_series.has_whole(forward_series.load(result.trial_id))
    again = forward_series.rebuild(session, every=True, spec_root=tmp_path / "specs")
    assert (spec.idea_id, result.trial_id, "уже есть") in again


def test_a_spec_edited_since_the_run_is_not_used_to_replay_it(session, tmp_path):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    _write_spec(spec.model_copy(update={"params": {"weight": 0.9}}), tmp_path / "specs")

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is None and "изменён" in replayed.reason


def test_a_changed_data_window_alone_does_not_hide_the_spec(session, tmp_path):
    """The night moves a spec's window to yesterday in the stored row, not in the file."""
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    older = spec.model_copy(update={"data": spec.data.model_copy(update={"end": date(2026, 1, 3)})})
    _write_spec(older, tmp_path / "specs")

    trial = session.get(Trial, result.trial_id)
    found = forward_series.spec_file_for(session, trial.spec_id, tmp_path / "specs")

    assert found == tmp_path / "specs" / f"{spec.idea_id}.yaml"


def test_a_run_judged_on_its_fit_period_replays_into_a_chart_without_a_forward_part(
        session, tmp_path, monkeypatch):
    spec = _explicit_long(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET)
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is not None, replayed.reason
    assert replayed.payload["judged_on_forward"] is False
    assert [p["part"] for p in replayed.payload["whole"]["parts"]] == ["selection"]
