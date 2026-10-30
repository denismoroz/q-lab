"""The ladder of an idea (`qlab.ladder`, docs/LADDER.md): seven steps for every idea, each with an
outcome and, for a step not run, a reason from a closed list. The three worked examples of
docs/EVALUATION_PATHS.md section 6 are built here on a synthetic registry, and so is one case per
reason."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab import ladder as lad
from qlab.ladder import Ledgers, Outcome, Reason
from qlab.registry import repo
from qlab.registry.models import (
    AnomalyScan,
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    TrialRoute,
    TrialSource,
    TrialStatus,
    VerdictStage,
)

RETUNE = "qlab.strategies.retune:Retune"


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def idea(session, idea_id: str, status: IdeaStatus = IdeaStatus.BENCH, parent: str | None = None,
         stage2: bool = False) -> None:
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     parent_id=parent,
                     notes=f"Second stage of {parent} (docs/TWO_STAGE.md)" if stage2 else None))
    session.flush()


def run(session, idea_id: str, route: TrialRoute, source: str = "binance", *,
        metrics: dict | None = None, reason: str = "r", code_ref: str = "x:Y",
        rules: str = "2026-10-29.2", end: date = date(2026, 10, 27)) -> int:
    spec = repo.add_spec(session, idea_id=idea_id, version=1, params={},
                         data_requirements={"source": source}, rebalance="1d", costs_model={},
                         code_ref=code_ref)
    trial = repo.add_trial(session, spec_id=spec.id, config_hash=f"h-{idea_id}", code_sha="x",
                           params={}, status=TrialStatus.OK, route=route, route_reason=reason,
                           metrics=metrics or {}, source=TrialSource.IMPORTED,
                           stand_version="2026-10-29.2",
                           started_at=datetime(2026, 10, 28, 2, tzinfo=UTC))
    repo.add_verdicts(session, [dict(
        idea_id=idea_id, spec_id=spec.id, trial_id=trial.id, stage=VerdictStage.EDGE,
        rule_id="net_edge_positive", rules_version=rules, metric="ann_return_net", value=0.1,
        comparator=">", threshold=0.0, passed=True, data_range_start=date(2025, 1, 1),
        data_range_end=end)])
    return trial.id


def ladder(session, idea_id: str, **ledgers):
    found = lad.build_ladder(session, idea_id, Ledgers(**ledgers))
    assert found is not None
    return found


def outcomes(found) -> list[str]:
    return [s.outcome.value for s in found.steps]


# --------------------------------------------------------------------------
# The shape every ladder has


def test_every_idea_has_the_same_seven_steps_and_a_reason_for_every_step_not_run(session) -> None:
    idea(session, "bare", IdeaStatus.CANDIDATE)
    idea(session, "done", IdeaStatus.VALIDATED)
    run(session, "done", TrialRoute.PAPER)

    for name in ("bare", "done"):
        found = ladder(session, name)
        assert [s.number for s in found.steps] == [1, 2, 3, 4, 5, 6, 7]
        assert [s.key for s in found.steps] == list(lad.STEP_KEYS)
        for step in found.steps:  # no silence: a step not run says why, and only such a step does
            assert (step.outcome == Outcome.NOT_RUN) == (step.reason is not None)
            assert step.title and step.outcome_ru


def test_an_unknown_idea_has_no_ladder(session) -> None:
    assert lad.build_ladder(session, "nope", Ledgers()) is None


def test_every_reason_has_a_russian_label_and_the_closed_list_is_the_documented_one() -> None:
    assert set(lad.REASON_RU) == set(Reason)
    assert all(lad.REASON_RU[r] for r in Reason)
    owner_nine = {
        "нет версии на Binance", "выбирать числам не из чего", "данные слишком тяжелы за ночь",
        "первая ступень не решила — сверять не с чем", "деньги стратегии не заданы",
        "идея закрыта до этой ступени", "ждёт очереди ночи", "карточка без реализации",
        "нет цен выбывших инструментов"}
    assert owner_nine <= set(lad.REASON_RU.values())


# --------------------------------------------------------------------------
# The three worked examples of docs/EVALUATION_PATHS.md, section 6


def test_example_bv2_research_headroom_causal(session) -> None:
    idea(session, "bv2-research-headroom-causal", IdeaStatus.VALIDATED)
    idea(session, "bv2-research-headroom-causal-hl", IdeaStatus.VALIDATED,
         parent="bv2-research-headroom-causal", stage2=True)
    first = run(session, "bv2-research-headroom-causal", TrialRoute.PAPER, "binance",
                metrics={"noise_cap_return_percentile": 0.99, "noise_cap_power": 0.27,
                         "forward_days": 513.0})
    second = run(session, "bv2-research-headroom-causal-hl", TrialRoute.PAPER, "hyperliquid",
                 metrics={"noise_cap_return_percentile": 0.99, "noise_cap_power": 0.10},
                 end=date(2026, 10, 28))

    found = ladder(session, "bv2-research-headroom-causal")

    assert outcomes(found) == ["passed", "passed", "passed", "not_run", "passed", "not_run",
                               "undecided"]
    step2, step3, step4, step6, step7 = (found.steps[i] for i in (1, 2, 3, 5, 6))
    assert step2.run.trial_id == first and "99%" in step2.detail and "27%" in step2.detail
    assert "по самой границе" in step2.detail  # passed at the very edge of a weak bar
    assert step3.run.trial_id == second and step3.run.venue == "hyperliquid"
    assert step3.other_idea == "bv2-research-headroom-causal-hl" and "10%" in step3.detail
    assert step3.run.data_end == "2026-10-28" and step3.run.rules_version == "2026-10-29.2"
    assert step3.run.stand_version == "2026-10-29.2"
    assert (step4.reason, step6.reason) == (Reason.NO_STRATEGY_MONEY, Reason.NOT_IN_PAPER)
    assert step7.outcome_ru == "ждёт владельца" and "владельца" in step7.detail


def test_example_xsmom_nobull_causal_ov_enter(session) -> None:
    idea(session, "xsmom-causal-ov-enter", IdeaStatus.RETIRED)
    run(session, "xsmom-causal-ov-enter", TrialRoute.PAPER, "binance",
        metrics={"noise_cap_return_percentile": 1.0, "noise_cap_power": 0.06})
    repo.set_status(session, idea_id="xsmom-causal-ov-enter", new_status=IdeaStatus.RETIRED,
                    reason="Owner 2026-10-28: «1, 2, 3 делай» -- overlays only on clean bases")
    session.add(AnomalyScan(idea_id="xsmom-causal-ov-enter", trial_id=1, config_version="v",
                            scanned_at=datetime(2026, 10, 29, tzinfo=UTC), route="paper",
                            fired=["a", "b", "c", "d", "e", "f"], flags=[], facts={}))
    session.flush()

    found = ladder(session, "xsmom-causal-ov-enter",
                   overlays={"xsmom-causal-ov-enter": {"stage2_pending": True}})

    assert outcomes(found) == ["passed", "passed", "not_run", "not_run", "passed", "not_run",
                               "failed"]
    assert [found.steps[i].reason for i in (2, 3, 5)] == [Reason.CLOSED_BEFORE] * 3
    assert "сработало признаков: 6" in found.steps[4].detail
    assert found.steps[4].outcome_ru == "записаны"
    assert "«1, 2, 3 делай»" in found.steps[6].detail  # the owner's own words, from the ledger


def test_example_frab_binance_spot_research_k3(session) -> None:
    idea(session, "frab-k3")
    run(session, "frab-k3", TrialRoute.NEEDS_FORWARD, "hyperliquid",
        metrics={"forward_days": 17.0, "forward_days_needed": 108.0, "noise_cap_power": 0.02,
                 "noise_cap_return_percentile": 0.46},
        reason="forward test too short to tell either way")
    heavy = {"frab-k3": {"day": "2026-10-28", "reason": (
        "the twin needs Binance's hourly candles for every listed perpetual since 2019; ... "
        "or a hand run with --allow-heavy")}}

    found = ladder(session, "frab-k3", twins=heavy)

    assert outcomes(found) == ["passed", "not_run", "undecided", "not_run", "passed", "not_run",
                               "not_run"]
    assert found.steps[1].reason == Reason.HEAVY_DATA
    assert "17 дн. из 108 дн." in found.steps[2].detail and "2%" in found.steps[2].detail
    assert found.steps[3].reason == Reason.NO_STRATEGY_MONEY
    assert "годы не определены" in found.steps[4].detail
    assert [found.steps[i].reason for i in (5, 6)] == [Reason.NOT_REACHED] * 2


# --------------------------------------------------------------------------
# One case per reason


def test_reason_no_binance_version_for_a_code_chosen_idea_on_hyperliquid(session) -> None:
    idea(session, "retune-hl")
    run(session, "retune-hl", TrialRoute.NEEDS_FORWARD, "hyperliquid", code_ref=RETUNE)

    assert ladder(session, "retune-hl").steps[1].reason == Reason.NO_BINANCE_VERSION


@pytest.mark.parametrize("words,reason", [
    ("the numbers were picked by hand and there is no declared alternative", Reason.NO_CHOICE),
    ("a and b differ in include_spot, instruments, source, not in numbers alone",
     Reason.NO_CHOICE),
    ("the twin carries 60 instruments and its hourly candles take 25 minutes", Reason.HEAVY_DATA),
    ("x names PURR in params.universe, which Binance does not list", Reason.NO_BINANCE_VERSION),
    ("the data comes from xyz, which has no Binance counterpart in the lab",
     Reason.NO_BINANCE_VERSION),
    ("something the table has never seen", Reason.TWIN_REFUSED_OTHER),
])
def test_the_twin_ledgers_words_become_a_closed_reason(session, words, reason) -> None:
    assert lad.twin_reason(words) == reason
    idea(session, "hand")
    run(session, "hand", TrialRoute.NEEDS_FORWARD, "hyperliquid")

    step = ladder(session, "hand", twins={"hand": {"reason": words, "day": "2026-10-28"}}).steps[1]

    assert step.reason == reason and words[:20] in step.detail


def test_reason_first_step_undecided_when_the_first_stage_tested_nothing(session) -> None:
    idea(session, "xyz-like", IdeaStatus.CANDIDATE)
    run(session, "xyz-like", TrialRoute.NOT_EVALUABLE, "hyperliquid",
        reason="the mechanism is not expressed in the code")

    found = ladder(session, "xyz-like")

    assert found.steps[0].reason == Reason.NOT_EVALUABLE
    assert [found.steps[i].reason for i in (1, 2)] == [Reason.FIRST_STEP_UNDECIDED] * 2


def test_reason_first_step_undecided_when_the_first_stage_was_rejected(session) -> None:
    idea(session, "rej", IdeaStatus.BENCH)  # a live second stage of a first stage that failed
    run(session, "rej", TrialRoute.REJECT, "binance", reason="shape_aware_edge failed")

    step3 = ladder(session, "rej").steps[2]

    assert step3.reason == Reason.FIRST_STEP_UNDECIDED


def test_reason_no_strategy_money_and_the_money_that_exists(session) -> None:
    idea(session, "poor")
    idea(session, "rich")
    run(session, "poor", TrialRoute.SHELF, "hyperliquid", metrics={"min_capital_usd": 8122.0})
    run(session, "rich", TrialRoute.PAPER, "hyperliquid", metrics={"min_capital_usd": 83.0})

    shelf = ladder(session, "poor").steps[3]
    fits = ladder(session, "rich", money={"rich": 257.0}).steps[3]
    short = ladder(session, "poor", money={"poor": 3000.0}).steps[3]
    unmeasured = ladder(session, "rich", money={"rich": 100.0}).steps[3]

    assert shelf.reason == Reason.NO_STRATEGY_MONEY and "$8,122" in shelf.detail
    assert "сравнить не с чем" in shelf.detail  # no money in the spec: nothing to compare with
    assert fits.outcome == Outcome.PASSED and "$83" in fits.detail and "$257" in fits.detail
    assert short.outcome == Outcome.FAILED
    assert "нужно $8,122, своих $3,000" in short.detail
    assert unmeasured.outcome == Outcome.PASSED  # $83 <= $100


def test_money_without_a_measured_minimum_is_undecided(session) -> None:
    idea(session, "unmeasured")
    run(session, "unmeasured", TrialRoute.NEEDS_FORWARD, "hyperliquid")

    assert ladder(session, "unmeasured", money={"unmeasured": 120.0}).steps[3].outcome \
        == Outcome.UNDECIDED


def test_reason_closed_before_for_a_rejected_idea(session) -> None:
    idea(session, "dead", IdeaStatus.REJECTED)
    run(session, "dead", TrialRoute.REJECT, "binance")

    found = ladder(session, "dead")

    assert found.steps[1].outcome == Outcome.FAILED
    assert [found.steps[i].reason for i in (2, 3, 5, 6)] == [Reason.CLOSED_BEFORE] * 4


def test_reason_waits_for_night_for_a_spec_with_no_run_and_for_a_twin_not_yet_built(
        session) -> None:
    idea(session, "queued", IdeaStatus.CANDIDATE)
    repo.add_spec(session, idea_id="queued", version=1, params={}, data_requirements={},
                  rebalance="1d", costs_model={}, code_ref="x:Y")
    idea(session, "file-only", IdeaStatus.CANDIDATE)
    idea(session, "waiting")
    run(session, "waiting", TrialRoute.NEEDS_FORWARD, "hyperliquid")
    idea(session, "deferred")
    run(session, "deferred", TrialRoute.NEEDS_FORWARD, "hyperliquid")

    assert [s.reason for s in ladder(session, "queued").steps[:3]] == [Reason.WAITS_FOR_NIGHT] * 3
    assert ladder(session, "file-only", spec_stems=frozenset({"file-only"})).steps[0].reason \
        == Reason.WAITS_FOR_NIGHT
    assert ladder(session, "waiting").steps[1].reason == Reason.WAITS_FOR_NIGHT
    cut = ladder(session, "deferred", deferred_twins=frozenset({"deferred"})).steps[1]
    assert cut.reason == Reason.WAITS_FOR_NIGHT and "предел" in cut.detail


def test_reason_card_without_implementation_and_a_card_in_the_queue(session) -> None:
    idea(session, "card", IdeaStatus.CANDIDATE)
    idea(session, "card-in-queue", IdeaStatus.CANDIDATE)

    bare = ladder(session, "card")
    queued = ladder(session, "card-in-queue", queue=frozenset({"card-in-queue"}))

    assert [s.reason for s in bare.steps[:3]] == [Reason.CARD_WITHOUT_IMPLEMENTATION] * 3
    assert bare.steps[4].reason == Reason.CARD_WITHOUT_IMPLEMENTATION
    assert queued.steps[0].reason == Reason.WAITS_FOR_NIGHT


def test_reason_no_delisted_prices(session) -> None:
    idea(session, "xyz", IdeaStatus.CANDIDATE)
    run(session, "xyz", TrialRoute.NOT_EVALUABLE, "hyperliquid",
        reason="the source serves no price history for 22 delisted instrument(s) of this "
               "universe, so only survivors could be tested: xyz:ACN")

    step = ladder(session, "xyz").steps[0]

    assert step.reason == Reason.NO_DELISTED_PRICES and "xyz:ACN" in step.detail


def test_reason_id_taken_for_a_binance_idea_whose_second_stage_name_belongs_to_another(
        session) -> None:
    idea(session, "pool")
    idea(session, "pool-hl", parent="pool")  # an experiment, not the second stage
    run(session, "pool", TrialRoute.NEEDS_FORWARD, "binance")

    assert ladder(session, "pool").steps[2].reason == Reason.ID_TAKEN


def test_a_binance_idea_without_a_second_stage_yet_waits_for_the_night(session) -> None:
    idea(session, "fresh")
    run(session, "fresh", TrialRoute.PAPER, "binance")

    step = ladder(session, "fresh", overlays={"fresh": {"stage2_pending": True}}).steps[2]

    assert step.reason == Reason.WAITS_FOR_NIGHT and "в долгу" in step.detail


def test_reasons_in_production_and_twin_only_while_waiting(session) -> None:
    idea(session, "book", IdeaStatus.PAPER)
    run(session, "book", TrialRoute.NEEDS_FORWARD, "hyperliquid")
    idea(session, "decided", IdeaStatus.BENCH)
    run(session, "decided", TrialRoute.SHELF, "hyperliquid")

    assert ladder(session, "book").steps[1].reason == Reason.IN_PRODUCTION
    assert ladder(session, "decided").steps[1].reason == Reason.TWIN_ONLY_WHILE_WAITING


# --------------------------------------------------------------------------
# Which run stands behind which step


def test_a_hyperliquid_idea_takes_step_two_from_its_causal_twin_and_names_its_second_stage(
        session) -> None:
    idea(session, "orig")
    run(session, "orig", TrialRoute.NEEDS_FORWARD, "hyperliquid")
    idea(session, "orig-causal", parent="orig")
    idea(session, "orig-causal-hl", parent="orig-causal", stage2=True)
    twin = run(session, "orig-causal", TrialRoute.PAPER, "binance",
               metrics={"noise_cap_return_percentile": 1.0, "noise_cap_power": 0.9})
    run(session, "orig-causal-hl", TrialRoute.SHELF, "hyperliquid")

    step2 = ladder(session, "orig").steps[1]

    assert step2.outcome == Outcome.PASSED and step2.run.trial_id == twin
    assert step2.other_idea == "orig-causal" and "двойник «orig-causal»" in step2.detail
    assert "orig-causal-hl" in step2.detail  # the twin's second stage is named


def test_a_second_stage_shows_its_first_stage_as_step_two(session) -> None:
    idea(session, "first", IdeaStatus.BENCH)
    idea(session, "first-hl", IdeaStatus.BENCH, parent="first", stage2=True)
    run(session, "first", TrialRoute.NEEDS_FORWARD, "binance")
    own = run(session, "first-hl", TrialRoute.PAPER, "hyperliquid",
              metrics={"stage1_trial_id": 7.0, "noise_cap_power": 0.1,
                       "noise_cap_return_percentile": 0.5})

    found = ladder(session, "first-hl")

    assert found.steps[1].other_idea == "first" and found.steps[1].outcome == Outcome.UNDECIDED
    assert found.steps[2].run.trial_id == own and "прогон 7" in found.steps[2].detail


def test_a_retune_binance_version_counts_as_the_binance_version(session) -> None:
    idea(session, "headroom-retune")
    run(session, "headroom-retune", TrialRoute.NEEDS_FORWARD, "hyperliquid", code_ref=RETUNE)
    idea(session, "headroom-retune-binance", parent="headroom-retune")
    run(session, "headroom-retune-binance", TrialRoute.NEEDS_FORWARD, "binance")

    step2 = ladder(session, "headroom-retune").steps[1]

    assert step2.other_idea == "headroom-retune-binance" and step2.outcome == Outcome.UNDECIDED


# --------------------------------------------------------------------------
# Step 6 and 7


def test_the_owners_gate_reads_the_status(session) -> None:
    for name, status in (("a", IdeaStatus.PAPER), ("b", IdeaStatus.LIVE),
                         ("c", IdeaStatus.VALIDATED), ("d", IdeaStatus.RETIRED)):
        idea(session, name, status)
        run(session, name, TrialRoute.PAPER, "hyperliquid")

    got = {n: ladder(session, n).steps[6] for n in "abcd"}

    assert got["a"].outcome == got["b"].outcome == Outcome.PASSED
    assert got["c"].outcome == Outcome.UNDECIDED
    assert got["d"].outcome == Outcome.FAILED


def test_a_live_book_has_no_paper_step(session) -> None:
    idea(session, "frab-like", IdeaStatus.LIVE)
    run(session, "frab-like", TrialRoute.NEEDS_FORWARD, "hyperliquid")
    books = {"frab-like": {"book": "x", "equity_from": "strategy_pnl", "strategy_capital_usd": 120,
                           "paper_file": "f.csv"}}

    step6 = ladder(session, "frab-like", paper_books=books).steps[5]

    assert step6.reason == Reason.NOT_IN_PAPER and "живая книга" in step6.detail


def test_load_ledgers_reads_the_files_and_survives_missing_ones(tmp_path) -> None:
    assert lad.load_ledgers(tmp_path) == Ledgers(root=tmp_path)  # nothing there: all empty
    (tmp_path / "night").mkdir()
    (tmp_path / "data" / "twins").mkdir(parents=True)
    (tmp_path / "data" / "twins" / "ledger.json").write_text('{"a": {"reason": "r"}}')
    (tmp_path / "data" / "night").mkdir()
    (tmp_path / "data" / "night" / "2026-10-29.json").write_text(
        '{"twins": {"deferred": ["x"]}}')
    (tmp_path / "specs" / "deep").mkdir(parents=True)
    (tmp_path / "specs" / "deep" / "an-idea.yaml").write_text("x: 1")
    (tmp_path / "night" / "implement_queue.yaml").write_text("cards:\n  - cards/c1.yaml\n")
    (tmp_path / "specs" / "a.yaml").write_text(
        "idea_id: b\ntitle: x\nmin_leg_notional: 10.0\nstrategy_capital_usd: 257\n")
    (tmp_path / "specs" / "deep" / "l.yaml").write_text(
        "idea_id: 'l'\nstrategy_capital_usd: 120.0  # the budget\n")
    (tmp_path / "specs" / "deep" / "two-a.yaml").write_text(
        "idea_id: two\nstrategy_capital_usd: 100\n")
    (tmp_path / "specs" / "deep" / "two-b.yaml").write_text(
        "idea_id: two\nstrategy_capital_usd: 200\n")  # files that disagree: no sum
    (tmp_path / "night" / "stage2.yaml").write_text("stage1_survives: [paper]\n")

    got = lad.load_ledgers(tmp_path)

    assert got.twins == {"a": {"reason": "r"}} and got.deferred_twins == {"x"}
    assert got.queue == {"c1"} and got.money == {"b": 257.0, "l": 120.0}
    assert got.spec_stems == {"an-idea", "a", "l", "two-a", "two-b"}
    assert got.stage1_survives == {"paper"}


# --------------------------------------------------------------------------
# The morning report


def test_the_report_table_has_a_row_per_open_idea_and_counts_the_reasons(session) -> None:
    idea(session, "live-one", IdeaStatus.LIVE)
    run(session, "live-one", TrialRoute.NEEDS_FORWARD, "hyperliquid")
    idea(session, "bench-one")
    run(session, "bench-one", TrialRoute.PAPER, "binance")
    idea(session, "card-one", IdeaStatus.CANDIDATE)
    idea(session, "gone", IdeaStatus.REJECTED)
    idea(session, "noise-dollar_neutral-1", IdeaStatus.CANDIDATE)

    assert lad.open_idea_ids(session) == ["live-one", "bench-one", "card-one"]
    text = "\n".join(lad.report_lines(session, Ledgers()))

    # the money of step 4 is one symbol per venue (Binance / Hyperliquid), then who passed
    assert ("| live-one | live | ✓ | · | ~ | ·/· | ✓ | · | ✓ | "
            "ни одна площадка не пройдена: на Binance — не проводилась, на Hyperliquid — рано "
            "судить |") in text
    assert "| card-one | candidate | · | · | · | ·/· | · | · | · | " in text
    assert "2 Binance, 3 Hyperliquid" in text
    assert "gone" not in text and "noise-" not in text
    assert "Не проводилось, по причинам" in text and "карточка без реализации" in text


def test_the_night_report_carries_the_ladder_section_when_the_stage_made_one(
        tmp_path, monkeypatch) -> None:
    import qlab.night as night

    monkeypatch.setattr(night, "REPORT_DIR", tmp_path)
    state = {"results": [], "ladder": ["| идея | статус |", "| a | live |"]}

    text = night.write_report(date(2026, 10, 29), state).read_text(encoding="utf-8")

    assert "## Лестница открытых идей" in text and "| a | live |" in text
    quiet = night.write_report(date(2026, 10, 30), {"results": []}).read_text(encoding="utf-8")
    assert "Лестница" not in quiet


# --------------------------------------------------------------------------
# The venue split (docs/VENUE_SPLIT.md): one verdict and one money check per venue


def test_steps_2_and_3_are_the_two_venues_and_step_4_has_a_line_for_each(session) -> None:
    idea(session, "both", IdeaStatus.BENCH)
    run(session, "both", TrialRoute.NEEDS_INFRASTRUCTURE, "binance",
        metrics={"min_capital_usd": 400.0})
    idea(session, "both-hl", IdeaStatus.BENCH, parent="both", stage2=True)
    run(session, "both-hl", TrialRoute.REJECT, "hyperliquid", metrics={"min_capital_usd": 90.0})

    found = ladder(session, "both", money={"both": 257.0})

    binance, hl = found.steps[1], found.steps[2]
    assert (binance.venue, hl.venue) == ("binance", "hyperliquid")
    assert (binance.key, hl.key) == ("binance", "hyperliquid")
    assert binance.outcome == Outcome.PASSED and "адаптера заявок на ней нет" in binance.detail
    assert "адаптера заявок на этой площадке у нас нет" in binance.execution_ru
    assert "frab на 10.8.0.5" in hl.execution_ru
    assert hl.outcome == Outcome.FAILED
    # each venue has its own money check: Binance needs $400 against $257, Hyperliquid $90
    by = {m.venue: m for m in found.steps[3].money}
    assert by["binance"].outcome == Outcome.FAILED
    assert "нужно $400, своих $257" in by["binance"].detail
    assert by["hyperliquid"].outcome == Outcome.PASSED and "$90" in by["hyperliquid"].detail
    # the step leads with the venue we can execute on
    assert found.steps[3].outcome == Outcome.PASSED
    # passing on one venue and failing on the other is a result, named as such
    assert found.summary_ru.startswith("прошла на Binance, на Hyperliquid — не прошла")
    assert "нет адаптера" in found.summary_ru
    assert [v.venue for v in found.venues] == ["binance", "hyperliquid"]
    body = found.to_dict()
    assert body["venues"][0]["word_ru"] == "пройден" and body["venues"][1]["route"] == "reject"
    assert body["steps"][3]["money"][0]["venue"] == "binance"


def test_an_idea_that_exists_on_one_venue_shows_the_other_as_not_run_with_a_reason(session) -> None:
    idea(session, "hl-only", IdeaStatus.BENCH)
    run(session, "hl-only", TrialRoute.PAPER, "hyperliquid", metrics={"min_capital_usd": 50.0})

    found = ladder(session, "hl-only", money={"hl-only": 120.0})

    assert found.steps[1].outcome == Outcome.NOT_RUN and found.steps[1].reason is not None
    by = {m.venue: m for m in found.steps[3].money}
    assert by["binance"].outcome == Outcome.NOT_RUN
    assert by["binance"].reason == Reason.NO_VERSION_ON_VENUE
    assert by["binance"].reason_ru == "нет версии на этой площадке"
    assert by["hyperliquid"].outcome == Outcome.PASSED
    assert found.summary_ru.startswith("прошла на Hyperliquid, на Binance — не проводилась")
    for step in found.steps:  # no silence: a step not run says why, and only such a step does
        assert (step.outcome == Outcome.NOT_RUN) == (step.reason is not None)


def test_a_rejected_binance_run_and_a_missing_second_stage_pass_nowhere(session) -> None:
    idea(session, "nowhere", IdeaStatus.REJECTED)
    run(session, "nowhere", TrialRoute.REJECT, "binance")

    found = ladder(session, "nowhere", money={"nowhere": 100.0})

    assert found.summary_ru.startswith("ни одна площадка не пройдена: на Binance — не прошла")
    by = {m.venue: m for m in found.steps[3].money}
    assert by["hyperliquid"].reason == Reason.CLOSED_BEFORE  # no second stage for a closed idea
    # the run did not measure the capital it needs
    assert by["binance"].outcome == Outcome.UNDECIDED
