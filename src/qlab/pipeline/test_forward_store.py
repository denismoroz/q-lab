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

    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)

    assert result.routing.route == "paper"
    stored = forward_series.load(result.trial_id)
    assert stored is not None and stored["built_by"] == "evaluate"
    assert stored["idea_id"] == spec.idea_id and stored["venue"] == "hyperliquid"
    # The same number the rules saw: the chart ends where the verdict's metrics do.
    assert stored["total_return"] == pytest.approx(result.metrics["total_return_net"])
    assert stored["matches_trial"] is True
    assert stored["points"][0][1] == 0.0 and len(stored["points"]) > 2
    assert stored["start"] == "2026-01-03"


def test_a_run_that_does_not_keep_its_idea_alive_stores_nothing(session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=_split_ruleset([CAPITAL_FIT_TIGHT]),
                           deployable_capital_usd=1000.0)

    assert result.routing.route == "reject"
    assert forward_series.load(result.trial_id) is None


def test_calibration_noise_stores_nothing(session, tmp_path):
    spec = _causal_spec(session, tmp_path)

    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0,
                           update_idea_status=False)

    assert forward_series.load(result.trial_id) is None


def test_a_run_judged_on_its_fit_period_stores_nothing(session, tmp_path):
    spec = _explicit_long(session, tmp_path)  # no fixed date: the whole window is the fit period

    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)

    assert result.metrics.get("judged_on_forward") in (None, 0.0)
    assert forward_series.load(result.trial_id) is None


def test_a_failure_to_write_the_series_does_not_fail_the_run(session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)

    def _boom(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(forward_series, "store_from_run", _boom)

    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)

    assert result.routing.route == "paper" and result.error is None
    assert session.get(Trial, result.trial_id) is not None


def _write_spec(spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{spec.idea_id}.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")


def test_a_stored_run_is_replayed_from_its_spec_and_snapshot_to_the_same_series(
        session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)
    live = forward_series.load(result.trial_id)
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is not None, replayed.reason
    assert replayed.payload["built_by"] == "rebuild" and replayed.payload["matches_trial"] is True
    assert replayed.payload["points"] == live["points"]


def test_rebuild_writes_the_series_a_run_made_before_this_existed_lacks(
        session, tmp_path, monkeypatch):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)
    forward_series.series_path(result.trial_id).unlink()  # a run from before the series was kept
    monkeypatch.setattr("qlab.rules.loader.load", lambda version: RULESET)
    _write_spec(spec, tmp_path / "specs")

    done = forward_series.rebuild(session, [spec.idea_id], spec_root=tmp_path / "specs")

    assert done == [(spec.idea_id, result.trial_id, "построен")]
    assert forward_series.load(result.trial_id)["built_by"] == "rebuild"


def test_a_spec_edited_since_the_run_is_not_used_to_replay_it(session, tmp_path):
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)
    _write_spec(spec.model_copy(update={"params": {"weight": 0.9}}), tmp_path / "specs")

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is None and "изменён" in replayed.reason


def test_a_changed_data_window_alone_does_not_hide_the_spec(session, tmp_path):
    """The night moves a spec's window to yesterday in the stored row, not in the file."""
    spec = _causal_spec(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)
    older = spec.model_copy(update={"data": spec.data.model_copy(update={"end": date(2026, 1, 3)})})
    _write_spec(older, tmp_path / "specs")

    trial = session.get(Trial, result.trial_id)
    found = forward_series.spec_file_for(session, trial.spec_id, tmp_path / "specs")

    assert found == tmp_path / "specs" / f"{spec.idea_id}.yaml"


def test_a_run_judged_on_its_fit_period_cannot_be_replayed_into_a_forward_chart(session, tmp_path):
    spec = _explicit_long(session, tmp_path)
    result = evaluate_spec(spec, session=session, ruleset=RULESET, deployable_capital_usd=1000.0)

    replayed = forward_series.replay(session, result.trial_id, spec_root=tmp_path / "specs")

    assert replayed.payload is None and "нет проверки вперёд" in replayed.reason
