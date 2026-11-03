"""The three labels «скамейка» stands for (`qlab.bench_label`, owner 2026-11-03): each of the three,
the order they are read in, and the idea that fits none ("причина не определена"), on bare venue
answers and on a synthetic registry."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import bench_label as bl
from qlab import ladder as lad
from qlab.bench_label import BenchReason, VenueAnswer
from qlab.ladder import Ledgers, Outcome
from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    Spec,
    TrialRoute,
    TrialSource,
    TrialStatus,
    VerdictStage,
)

P, F, U, N = Outcome.PASSED, Outcome.FAILED, Outcome.UNDECIDED, Outcome.NOT_RUN


def answers(binance, hl):
    """Each side is `(outcome, route)` or `(outcome, route, noise_power)`."""
    def one(venue, side):
        outcome, route, *power = side
        return VenueAnswer(venue, outcome, route, power[0] if power else None)

    return (one("binance", binance), one("hyperliquid", hl))


# --------------------------------------------------------------------------
# The reading, on bare answers


def test_passed_on_binance_with_no_executor_is_passed_no_execution() -> None:
    got = bl.bench_reason(answers((P, "needs-infrastructure"), (F, "reject")))
    assert got == BenchReason.PASSED_NO_EXECUTION
    assert bl.bench_reason(answers((P, "needs-infrastructure"), (N, None))) == got


def test_a_binance_pass_outranks_a_waiting_hyperliquid() -> None:
    # the pass is the fact the owner needs first; the waiting venue could still add execution
    got = bl.bench_reason(answers((P, "needs-infrastructure"), (U, "needs-forward")))
    assert got == BenchReason.PASSED_NO_EXECUTION


def test_one_venue_rejected_and_the_other_silent_waits_for_the_second_venue() -> None:
    for silent in ((U, "needs-forward"), (U, "needs-more-data"), (N, None), (U, "error")):
        assert bl.bench_reason(answers((F, "reject"), silent)) == BenchReason.AWAITS_SECOND_VENUE
        assert bl.bench_reason(answers(silent, (F, "reject"))) == BenchReason.AWAITS_SECOND_VENUE


def test_waiting_for_the_second_venue_outranks_too_early() -> None:
    # the other venue's answer is «рано судить» too, but a rejection plus a missing answer is the
    # more specific state
    got = bl.bench_reason(answers((F, "reject"), (U, "needs-forward", 0.1)))
    assert got == BenchReason.AWAITS_SECOND_VENUE


def test_a_venue_that_answered_needs_forward_or_more_data_is_too_early() -> None:
    assert bl.bench_reason(answers((U, "needs-forward"), (U, "needs-forward"))) == (
        BenchReason.TOO_EARLY)
    assert bl.bench_reason(answers((N, None), (U, "needs-forward"))) == BenchReason.TOO_EARLY
    assert bl.bench_reason(answers((U, "needs-more-data"), (N, None))) == BenchReason.TOO_EARLY


def test_a_noise_bar_without_power_is_too_early_whatever_the_route() -> None:
    assert bl.bench_reason(answers((U, "error", 0.5), (N, None))) == BenchReason.TOO_EARLY
    # at or above the line the bar had power: an undecided crash is not «рано судить»
    assert bl.bench_reason(answers((U, "error", bl.NO_POWER_BELOW), (N, None))) == (
        BenchReason.UNDETERMINED)


@pytest.mark.parametrize("pair", [
    answers((N, None), (N, None)),                                # nothing ran
    answers((F, "reject"), (F, "reject")),                        # both rejected: not a bench idea
    answers((P, "needs-infrastructure"), (P, "paper")),           # both passed: validated
    answers((N, None), (P, "paper")),                             # a Hyperliquid pass on the bench
    answers((U, "error"), (N, None)),                             # a crash, no power measured
    answers((P, "shelf"), (N, None)),                             # an old-rules pass, nothing else
])
def test_an_idea_that_fits_none_says_so(pair) -> None:
    assert bl.bench_reason(pair) == BenchReason.UNDETERMINED


def test_every_label_has_words_a_hint_and_a_place_in_the_order() -> None:
    assert set(bl.BENCH_LABEL_RU) == set(BenchReason) == set(bl.BENCH_ORDER)
    assert set(bl.BENCH_HINT_RU) == set(BenchReason)
    assert [bl.BENCH_LABEL_RU[r] for r in bl.BENCH_ORDER] == [
        "рано судить", "прошла, нет исполнения", "ждёт вторую площадку",
        "скамейка — причина не определена"]


def test_the_page_block_carries_the_filter_key_and_the_words() -> None:
    block = bl.bench_block(BenchReason.PASSED_NO_EXECUTION)
    assert block is not None
    assert block["key"] == "bench-passed-no-execution"
    assert block["label_ru"] == "прошла, нет исполнения" and block["hint_ru"]
    assert bl.bench_block(None) is None
    assert bl.status_key("validated", None) == "validated"
    assert bl.status_key("bench", BenchReason.TOO_EARLY) == "bench-too-early"
    assert bl.status_key("paper", BenchReason.TOO_EARLY) == "paper"  # only the bench is split


# --------------------------------------------------------------------------
# On a registry: the ladder's venue views feed the reading


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _idea(session, idea_id, status=IdeaStatus.BENCH, parent=None, stage2=False) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if stage2 else None))
    session.flush()


def _run(session, idea_id, route, source, power=None) -> None:
    version = session.query(Spec).filter(Spec.idea_id == idea_id).count() + 1
    spec = repo.add_spec(session, idea_id=idea_id, version=version, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref="x:Y")
    trial = repo.add_trial(
        session, spec_id=spec.id, config_hash=f"h-{idea_id}-{version}", code_sha="x", params={},
        status=TrialStatus.OK, route=route, route_reason="r",
        metrics={} if power is None else {"noise_cap_power": power},
        source=TrialSource.IMPORTED, stand_version="2026-11-03.1",
        started_at=datetime(2026, 11, 2, 2, tzinfo=UTC))
    repo.add_verdicts(session, [dict(
        idea_id=idea_id, spec_id=spec.id, trial_id=trial.id, stage=VerdictStage.EDGE,
        rule_id="net_edge_positive", rules_version="2026-11-03.1", metric="ann_return_net",
        value=0.1, comparator=">", threshold=0.0, passed=True,
        data_range_start=date(2025, 1, 1), data_range_end=date(2026, 10, 27))])


def _pair(session, name, binance, hl, status=IdeaStatus.BENCH) -> None:
    """A Binance idea with its Hyperliquid second stage; `hl=None` leaves the second stage out."""
    _idea(session, name, status)
    _run(session, name, binance, "binance")
    if hl is not None:
        _idea(session, f"{name}-hl", status, parent=name, stage2=True)
        _run(session, f"{name}-hl", hl, "hyperliquid")


@pytest.fixture()
def registry(session):
    R = TrialRoute
    _pair(session, "early", R.NEEDS_FORWARD, R.NEEDS_FORWARD)
    _pair(session, "executionless", R.NEEDS_INFRASTRUCTURE, R.NEEDS_FORWARD)
    _pair(session, "waits", R.REJECT, R.NEEDS_FORWARD)
    _pair(session, "waits-other-way", R.NEEDS_FORWARD, R.REJECT)
    # no reason the code can name: Binance crashed, Hyperliquid never ran
    _pair(session, "crashed", R.ERROR, None)
    # not on the bench: no label, whatever the runs say
    _pair(session, "validated-one", R.NEEDS_INFRASTRUCTURE, R.PAPER, IdeaStatus.VALIDATED)
    # calibration noise is no idea of the lab's
    _idea(session, "noise-1")
    session.commit()
    return session


def test_the_labels_of_a_synthetic_registry(registry) -> None:
    got = bl.bench_reasons(registry, Ledgers())

    assert got["early"] == BenchReason.TOO_EARLY
    assert got["executionless"] == BenchReason.PASSED_NO_EXECUTION
    assert got["waits"] == BenchReason.AWAITS_SECOND_VENUE
    assert got["waits-other-way"] == BenchReason.AWAITS_SECOND_VENUE
    assert got["crashed"] == BenchReason.UNDETERMINED
    # second stages are ideas of the bench too, each read on its own ladder
    assert "early-hl" in got and "validated-one" not in got and "noise-1" not in got


def test_undetermined_ideas_are_listed_by_name(registry) -> None:
    got = bl.bench_reasons(registry, Ledgers())

    assert "crashed" in bl.undetermined_ideas(got)
    assert "early" not in bl.undetermined_ideas(got)
    assert bl.undetermined_ideas(got) == sorted(bl.undetermined_ideas(got))


def test_the_counts_keep_all_four_labels_in_order(registry) -> None:
    counts = bl.bench_counts(bl.bench_reasons(registry, Ledgers()))

    assert [c["reason"] for c in counts] == [r.value for r in bl.BENCH_ORDER]
    assert sum(c["count"] for c in counts) == len(bl.bench_reasons(registry, Ledgers()))
    assert all(c["label_ru"] and c["key"].startswith("bench-") for c in counts)


def test_a_ladder_off_the_bench_has_no_label(registry) -> None:
    found = lad.build_ladder(registry, "validated-one", Ledgers())
    assert found is not None and bl.bench_reason_of_ladder(found) is None


def test_the_morning_report_names_the_label_not_the_bare_status(registry) -> None:
    text = "\n".join(lad.report_lines(registry, Ledgers()))

    assert "| early | рано судить |" in text
    assert "| executionless | прошла, нет исполнения |" in text
    assert "| waits | ждёт вторую площадку |" in text
    assert "| crashed | скамейка — причина не определена |" in text
    assert "| validated-one | validated |" in text
    assert "Статус «скамейка» по надписям: рано судить" in text
