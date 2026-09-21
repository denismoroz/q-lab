"""Tests for `qlab.calibration.report` (docs/TASKS.md, T16).

Builds `Evaluation`/`NoiseTrial` objects directly rather than running the
real pipeline -- `report.py`'s job is pure summarisation and rendering, so
these tests isolate that from `evaluate_spec`/data concerns (covered by
`qlab.calibration.test_run`).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qlab.calibration.report import (
    Distribution,
    RealResult,
    calibration_report_path,
    render_report,
    summarize_by_generator,
    summarize_real,
    summarize_series,
    write_report,
)
from qlab.calibration.run import NoiseTrial
from qlab.pipeline.evaluate import Evaluation, RoutingDecision
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage


def _evaluation(
    *, trial_id: int, route: str, ann_return_net: float | None, sharpe_net: float | None, error=None
) -> Evaluation:
    metrics = None
    if error is None:
        metrics = {"ann_return_net": ann_return_net, "sharpe_net": sharpe_net}
    return Evaluation(
        trial_id=trial_id,
        metrics=metrics,
        rules_result=None,
        routing=RoutingDecision(route=route, reason="test"),
        error=error,
    )


def _trial(series, generator, seed, **kwargs) -> NoiseTrial:
    return NoiseTrial(
        series=series, generator=generator, seed=seed, evaluation=_evaluation(**kwargs)
    )


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------


def test_distribution_from_values_computes_quartiles():
    dist = Distribution.from_values([1.0, 2.0, 3.0, 4.0])
    assert dist.n == 4
    assert dist.min == 1.0
    assert dist.max == 4.0
    assert dist.median == pytest.approx(2.5)


def test_distribution_from_values_empty_is_none():
    assert Distribution.from_values([]) is None


# --------------------------------------------------------------------------
# summarize_series
# --------------------------------------------------------------------------


def test_summarize_series_counts_every_route():
    trials = [
        _trial(
            "dollar_neutral", "random_weights", 1,
            trial_id=1, route="paper", ann_return_net=0.10, sharpe_net=1.5,
        ),
        _trial(
            "dollar_neutral", "random_signs", 2,
            trial_id=2, route="shelf", ann_return_net=0.06, sharpe_net=0.8,
        ),
        _trial(
            "dollar_neutral", "shuffled_instruments", 3,
            trial_id=3, route="reject", ann_return_net=-0.02, sharpe_net=-0.5,
        ),
        # A "needs-more-data" trial still HAS metrics -- decide_route routes
        # here when some OTHER rule's metric is unknown (e.g. a preflight
        # fact), not because ann_return_net itself is missing.
        _trial(
            "dollar_neutral", "bootstrap_time", 4,
            trial_id=4, route="needs-more-data", ann_return_net=0.01, sharpe_net=0.2,
        ),
        _trial(
            "dollar_neutral", "random_weights", 5,
            trial_id=5, route="error", ann_return_net=None, sharpe_net=None, error="boom",
        ),
    ]

    summary = summarize_series(trials)

    assert summary.series == "dollar_neutral"
    assert summary.n_trials == 5
    assert summary.n_admitted == 2  # paper + shelf
    assert summary.n_rejected == 1
    assert summary.n_needs_more_data == 1
    assert summary.n_error == 1
    assert summary.admission_rate == pytest.approx(2 / 5)
    assert summary.ann_return_net.n == 4  # only the errored trial has no metrics at all
    assert summary.best_sharpe_net == pytest.approx(1.5)
    assert summary.best_sharpe_trial.seed == 1


def test_summarize_series_best_sharpe_ignores_nan():
    trials = [
        _trial(
            "unconstrained", "random_signs", 1,
            trial_id=1, route="reject", ann_return_net=-0.01, sharpe_net=float("nan"),
        ),
        _trial(
            "unconstrained", "random_signs", 2,
            trial_id=2, route="reject", ann_return_net=-0.02, sharpe_net=-0.3,
        ),
    ]
    summary = summarize_series(trials)
    assert summary.best_sharpe_net == pytest.approx(-0.3)


def test_summarize_series_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        summarize_series([])


def test_summarize_series_rejects_mixed_series():
    trials = [
        _trial(
            "dollar_neutral", "random_weights", 1,
            trial_id=1, route="reject", ann_return_net=-0.01, sharpe_net=-0.1,
        ),
        _trial(
            "unconstrained", "random_weights", 2,
            trial_id=2, route="reject", ann_return_net=-0.01, sharpe_net=-0.1,
        ),
    ]
    with pytest.raises(ValueError, match="same series"):
        summarize_series(trials)


# --------------------------------------------------------------------------
# summarize_by_generator
# --------------------------------------------------------------------------


def test_summarize_by_generator_splits_by_generator_within_one_series():
    trials = [
        _trial(
            "unconstrained", "bootstrap_time", 1,
            trial_id=1, route="paper", ann_return_net=0.50, sharpe_net=1.7,
        ),
        _trial(
            "unconstrained", "bootstrap_time", 2,
            trial_id=2, route="paper", ann_return_net=0.30, sharpe_net=1.2,
        ),
        _trial(
            "unconstrained", "random_signs", 3,
            trial_id=3, route="reject", ann_return_net=-0.02, sharpe_net=-0.5,
        ),
    ]

    breakdown = summarize_by_generator(trials)

    assert set(breakdown) == {"bootstrap_time", "random_signs"}
    assert breakdown["bootstrap_time"].n_trials == 2
    assert breakdown["bootstrap_time"].n_admitted == 2
    assert breakdown["bootstrap_time"].best_sharpe_net == pytest.approx(1.7)
    assert breakdown["random_signs"].n_trials == 1
    assert breakdown["random_signs"].n_admitted == 0
    # SeriesSummary.series still reports the real series, not the generator
    assert breakdown["bootstrap_time"].series == "unconstrained"


def test_render_report_generator_breakdown_flags_a_dominant_generator():
    """Regression scenario for the `bootstrap_time` look-ahead bug: one
    generator supplies (almost) all of a series' admissions. The report must
    say so by name, not just show an aggregate rate that hides it."""
    un_trials = (
        [
            _trial(
                "unconstrained", "bootstrap_time", i,
                trial_id=i, route="paper", ann_return_net=0.3, sharpe_net=1.5,
            )
            for i in range(1, 11)
        ]
        + [
            _trial(
                "unconstrained", "random_signs", 100 + i,
                trial_id=100 + i, route="reject", ann_return_net=-0.05, sharpe_net=-0.5,
            )
            for i in range(1, 11)
        ]
    )
    dn_trials = [
        _trial(
            "dollar_neutral", "random_signs", i,
            trial_id=200 + i, route="reject", ann_return_net=-0.03, sharpe_net=-0.4,
        )
        for i in range(1, 11)
    ]
    series_summaries = {
        "dollar_neutral": summarize_series(dn_trials),
        "unconstrained": summarize_series(un_trials),
    }
    generator_breakdowns = {
        "dollar_neutral": summarize_by_generator(dn_trials),
        "unconstrained": summarize_by_generator(un_trials),
    }

    content = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=[],
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
        generator_breakdowns=generator_breakdowns,
        include_revision_note=False,
    )

    assert "от одного генератора, `bootstrap_time`" in content
    assert "## Разбивка по генераторам" in content
    assert "`bootstrap_time`" in content
    assert "`random_signs`" in content


def test_render_report_include_revision_note_toggle():
    series_summaries = _sample_series_summaries()
    with_note = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=[],
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
        include_revision_note=True,
    )
    without_note = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=[],
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
        include_revision_note=False,
    )
    assert "Ревизия 2026-09-21" in with_note
    assert "Ревизия 2026-09-21" not in without_note


def test_render_report_main_result_section_compares_trend_to_best_noise():
    dn_trials = [
        _trial(
            "dollar_neutral", "shuffled_instruments", i,
            trial_id=i, route="reject",
            ann_return_net=-0.02 + 0.001 * i, sharpe_net=-0.5 + 0.01 * i,
        )
        for i in range(1, 51)
    ]
    series_summaries = {"dollar_neutral": summarize_series(dn_trials)}
    real_results = [
        RealResult(
            idea_id="trend-tsmom-crypto",
            title="TSMOM ансамбль",
            route="shelf",
            reason="passed but needs capital",
            ann_return_net=0.0475,
            sharpe_net=0.305,
            min_capital_usd=37829.0,
        )
    ]

    content = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=real_results,
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
        include_revision_note=False,
    )

    assert "Главный результат: trend против лучшего чистого шума" in content
    assert "0.305" in content


# --------------------------------------------------------------------------
# summarize_real / render_report
# --------------------------------------------------------------------------


def _ruleset() -> RuleSet:
    return RuleSet(
        version="2026-09-21.1",
        based_on=None,
        rules=[
            Rule(
                id="net_edge_positive",
                stage=Stage.EDGE,
                metric="ann_return_net",
                comparator=Comparator.GE,
                threshold=0.04,
                fatal=True,
            )
        ],
        retired=[],
    )


def test_summarize_real_reads_metrics_and_titles():
    evaluations = {
        "trend-tsmom-crypto": _evaluation(
            trial_id=1, route="shelf", ann_return_net=0.0475, sharpe_net=0.31
        )
    }
    rows = summarize_real(evaluations, {"trend-tsmom-crypto": "TSMOM ансамбль"})
    assert rows == [
        RealResult(
            idea_id="trend-tsmom-crypto",
            title="TSMOM ансамбль",
            route="shelf",
            reason="test",
            ann_return_net=0.0475,
            sharpe_net=0.31,
            min_capital_usd=None,
        )
    ]


def _sample_series_summaries():
    dn_trials = [
        _trial(
            "dollar_neutral",
            ["random_weights", "shuffled_instruments", "random_signs", "bootstrap_time"][i % 4],
            i,
            trial_id=i,
            route="paper" if i % 10 == 0 else "reject",
            ann_return_net=-0.05 + 0.001 * i,
            sharpe_net=-1.0 + 0.02 * i,
        )
        for i in range(1, 41)
    ]
    un_trials = [
        _trial(
            "unconstrained",
            ["random_weights", "shuffled_instruments", "random_signs", "bootstrap_time"][i % 4],
            i,
            trial_id=100 + i,
            route="paper" if i % 3 == 0 else "reject",
            ann_return_net=-0.02 + 0.01 * i,
            sharpe_net=-0.5 + 0.05 * i,
        )
        for i in range(1, 41)
    ]
    return {
        "dollar_neutral": summarize_series(dn_trials),
        "unconstrained": summarize_series(un_trials),
    }


def test_render_report_contains_headline_numbers_and_is_russian():
    series_summaries = _sample_series_summaries()
    real_results = [
        RealResult(
            idea_id="trend-tsmom-crypto",
            title="TSMOM ансамбль",
            route="shelf",
            reason="passed but needs capital",
            ann_return_net=0.0475,
            sharpe_net=0.305,
            min_capital_usd=37829.0,
        ),
        RealResult(
            idea_id="strategy-b-v2",
            title="Strategy B v2",
            route="reject",
            reason="net_edge_positive failed",
            ann_return_net=0.0074,
            sharpe_net=0.121,
            min_capital_usd=229.0,
        ),
    ]

    content = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=real_results,
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
    )

    assert "Калибровка правил" in content
    assert "2026-09-21.1" in content
    assert "dollar-neutral" in content
    assert "без ограничения на нетто" in content
    assert "trend-tsmom-crypto" in content
    assert "TSMOM ансамбль" in content
    assert f"{series_summaries['dollar_neutral'].n_trials}" in content
    assert "Оговорки" in content
    # No mojibake / undecoded escapes -- render_report should be plain text
    assert "\\n" not in content


def test_render_report_notes_the_headline_gap_when_unconstrained_passes_more():
    dn_trials = [
        _trial(
            "dollar_neutral", "random_weights", i,
            trial_id=i, route="reject", ann_return_net=-0.05, sharpe_net=-1.0,
        )
        for i in range(1, 21)
    ]
    un_trials = [
        _trial(
            "unconstrained",
            "random_weights",
            i,
            trial_id=100 + i,
            route="paper",
            ann_return_net=0.10,
            sharpe_net=1.2,
        )
        for i in range(1, 21)
    ]
    series_summaries = {
        "dollar_neutral": summarize_series(dn_trials),
        "unconstrained": summarize_series(un_trials),
    }

    content = render_report(
        ruleset=_ruleset(),
        series_summaries=series_summaries,
        real_results=[],
        reference_idea_id="trend-tsmom-crypto",
        generated_on=date(2026, 9, 21),
    )

    assert "заметно чаще" in content
    assert "рыночную бету и carry" in content


# --------------------------------------------------------------------------
# write_report / calibration_report_path
# --------------------------------------------------------------------------


def test_calibration_report_path_names_file_by_version(tmp_path: Path):
    path = calibration_report_path("2026-09-21.1", docs_dir=tmp_path)
    assert path == tmp_path / "CALIBRATION_2026-09-21.1.md"


def test_write_report_persists_content(tmp_path: Path):
    path = write_report("# hello\n", "2026-09-21.1", docs_dir=tmp_path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == "# hello\n"


def test_write_report_overwrites_same_version_only(tmp_path: Path):
    write_report("first\n", "2026-09-21.1", docs_dir=tmp_path)
    write_report("second\n", "2026-09-21.1", docs_dir=tmp_path)
    write_report("other\n", "2026-09-20.2", docs_dir=tmp_path)

    assert (tmp_path / "CALIBRATION_2026-09-21.1.md").read_text(encoding="utf-8") == "second\n"
    assert (tmp_path / "CALIBRATION_2026-09-20.2.md").read_text(encoding="utf-8") == "other\n"
