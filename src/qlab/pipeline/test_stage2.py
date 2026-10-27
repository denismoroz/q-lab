"""Tests for the second stage of the idea generator's evaluation (docs/TWO_STAGE.md):
the translation, the point-in-time mapping, the not-evaluable guard, the routing
(an idea needs both stages), the graveyard's data range and the absence of any agent."""

from __future__ import annotations

import dataclasses
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qlab.pipeline.sources as sources
import qlab.registry.db as db_module
from qlab.data.panel import MarketPanel
from qlab.pipeline import confirm, stage2
from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.registry import repo
from qlab.registry.lifecycle import apply_route
from qlab.registry.models import (
    AssetClass,
    Base,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
    StageTransition,
    Trial,
    TrialRoute,
    TrialStatus,
    Verdict,
)
from qlab.registry.stages import NOTES_PREFIX, child_id, stage2_child
from qlab.rules.schema import (
    Comparator,
    FitPeriodUse,
    HoldoutShare,
    NoisePower,
    Rule,
    RuleSet,
    ShortForwardUse,
    Stage,
    Stage2Confirmation,
)

PARENT = "gen-test-momentum"
CHILD = f"{PARENT}-hl"
SNAPSHOT = "snap-hl"
OWNER = "owner 2026-01-01: test fixture"

STAGE1_PARAMS = {
    "universe": {"rule": "all"},
    "signal": {"block": "momentum", "lookback_days": 14},
    "portfolio": {"block": "long_short_quantile", "quantiles": 5},
    "sizing": {"block": "equal", "gross": 1.0},
    "rebalance_days": 7,
    "min_history_days": 30,
}
FIRST_DAY = date(2025, 1, 1)
DAYS = 200
LAST_DAY = date(2025, 7, 19)  # FIRST_DAY + 199 days

CONFIG = stage2.Stage2Config(
    enabled=True, venue="hyperliquid", data_start=FIRST_DAY, taker_fee_bps=4.5,
    slippage_bps=0.9,
    cost_sources={"costs.taker_fee_bps": OWNER, "costs.slippage_bps": OWNER},
    stage1_survives=("paper", "shelf", "needs-infrastructure", "needs-forward"))

CAPITAL_FIT_GENEROUS = Rule(id="capital_fit", stage=Stage.PREFLIGHT, metric="min_capital_usd",
                            comparator=Comparator.LE, threshold=1_000_000, fatal=True)
CAPITAL_FIT_TIGHT = Rule(id="capital_fit", stage=Stage.PREFLIGHT, metric="min_capital_usd",
                         comparator=Comparator.LE, threshold=1.0, fatal=True)
PASSING = RuleSet(version="2026-01-01.1", rules=[CAPITAL_FIT_GENEROUS])
FAILING = RuleSet(version="2026-01-01.1", rules=[CAPITAL_FIT_TIGHT])


def _spec(**overrides: object) -> StrategySpec:
    base: dict[str, object] = {
        "idea_id": PARENT,
        "title": "Продолжение движения монет",
        "code_ref": "qlab.strategies.blocks:Composed",
        "params": STAGE1_PARAMS,
        "data": {"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                 "end": LAST_DAY},
        "costs": {"taker_fee_bps": 3.5, "slippage_bps": 0.9},
        "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [],
        "differs_from_source": ["Binance contracts, not composite prices."],
        "params_fixed_at": date(2025, 3, 1),
        "params_fixed_evidence": "A paper's sample ends 2025-02-28.",
        "sources": {k: OWNER for k in ("universe", "signal", "portfolio", "sizing",
                                        "rebalance_days", "min_history_days", "costs",
                                        "min_leg_notional")},
    }
    base.update(overrides)
    return StrategySpec.model_validate(base)


def _market(n_coins: int = 8, listed_at: dict[int, int] | None = None,
            seed: int = 3) -> MarketPanel:
    """`n_coins` perpetuals over DAYS days; coin i is listed from day `listed_at[i]` (0)."""
    index = pd.date_range(pd.Timestamp(FIRST_DAY, tz="UTC"), periods=DAYS, freq="1D")
    columns = [f"C{i}" for i in range(n_coins)]
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.001, 0.02, (DAYS, n_coins)), axis=0),
        index=index, columns=columns)
    for i, first in (listed_at or {}).items():
        prices.iloc[:first, i] = np.nan
    tradeable = prices.notna()
    funding = pd.DataFrame(0.0001, index=index, columns=columns).where(tradeable)
    volume = pd.DataFrame(1e6, index=index, columns=columns).where(tradeable)
    return MarketPanel(
        snapshot_id=SNAPSHOT, prices=prices, funding=funding, tradeable=tradeable,
        volume=volume,
        meta={"venue": "hyperliquid", "interval": "1d", "universe_complete": True,
              "range_start": index[0].isoformat(), "range_end": index[-1].isoformat()})


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """A throwaway registry as the global one (`session_scope` is what the step opens)."""
    monkeypatch.setenv("QLAB_DB", str(tmp_path / "qlab.db"))
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    Base.metadata.create_all(db_module.get_engine())
    with db_module.session_scope() as session:
        repo.upsert_idea(session, id=PARENT, title="Parent", source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                         notes="Generated from docs/MECHANISMS.md (m1)")
        repo.add_data_snapshot(session, id=SNAPSHOT, source="hyperliquid", instruments={},
                               range_start=FIRST_DAY, range_end=LAST_DAY, path="x", rows=DAYS)
    yield db_module
    db_module._engine = None
    db_module._SessionLocal = None


def _stand(monkeypatch, panel: MarketPanel, ruleset: RuleSet) -> None:
    """The step's collaborators: this panel instead of a fetch, this ruleset instead of the
    latest one, the candidate run without the 200 noise books (they test nothing here)."""
    monkeypatch.setattr("qlab.pipeline.evaluate.resolve_panel", lambda session, spec: panel)
    monkeypatch.setattr("qlab.rules.loader.load_latest", lambda: ruleset)

    def bar(spec, *, session, ruleset, deployable_capital_usd, n_trials):
        return types.SimpleNamespace(candidate=evaluate_spec(
            spec, session=session, ruleset=ruleset, deployable_capital_usd=deployable_capital_usd,
            check_lookahead=False))

    monkeypatch.setattr("qlab.calibration.shape_aware.evaluate_spec_with_shape_aware_bar", bar)


def _status(idea_id: str) -> IdeaStatus:
    with db_module.session_scope() as session:
        return session.get(Idea, idea_id).status


def _set_status(idea_id: str, status: IdeaStatus) -> None:
    with db_module.session_scope() as session:
        repo.set_status(session, idea_id=idea_id, new_status=status, reason="stage 1")


def _run(tmp_path: Path, route: str | None = "paper", **kw) -> stage2.Stage2Result:
    kw.setdefault("config", CONFIG)
    return stage2.run_stage2(_spec(), stage1_route=route, stage1_trial_id=None, noise_trials=1,
                             deployable_capital_usd=3000.0, spec_dir=tmp_path / "stage2", **kw)


# ---- the translation ------------------------------------------------------------------


def test_translation_keeps_every_parameter_value_and_switches_venue_and_costs() -> None:
    spec = _spec()
    child = stage2.translate(spec, CONFIG).spec

    assert child.params == spec.params and child.params is not spec.params  # a copy, untouched
    assert child.params["signal"]["lookback_days"] == 14 and child.params["rebalance_days"] == 7
    assert child.code_ref == spec.code_ref and child.data.interval == "1d"
    assert child.data.source == "hyperliquid" and spec.data.source == "binance"
    assert child.data.instruments is None  # the venue's own listing, discovered
    assert (child.costs.taker_fee_bps, child.costs.slippage_bps) == (4.5, 0.9)
    assert child.idea_id == CHILD and child.title.endswith("(второй этап: hyperliquid)")
    assert child.min_leg_notional == spec.min_leg_notional
    assert child.exploratory is None and child.judged_from is None


def test_translation_starts_the_data_at_the_later_of_the_two_starts() -> None:
    assert stage2.translate(_spec(), CONFIG).spec.data.start == FIRST_DAY
    late = _spec(data={"source": "binance", "interval": "1d", "start": date(2025, 3, 1),
                       "end": LAST_DAY})
    assert stage2.translate(late, CONFIG).spec.data.start == date(2025, 3, 1)
    assert stage2.translate(_spec(), CONFIG).spec.data.end == LAST_DAY


def test_translation_inherits_the_fix_date_so_the_venue_window_after_it_is_forward() -> None:
    spec = _spec(params_fit_from=date(2024, 1, 1))
    child = stage2.translate(spec, CONFIG).spec
    assert child.params_fixed_at == date(2025, 3, 1) and child.params_fit_from == date(2024, 1, 1)
    assert child.params_fixed_evidence.endswith(spec.params_fixed_evidence)
    assert "Inherited from stage 1" in child.params_fixed_evidence
    assert child.selects_causally == spec.selects_causally


def test_translation_says_what_it_did_and_cites_the_new_costs() -> None:
    spec = _spec()
    child = stage2.translate(spec, CONFIG).spec
    assert child.differs_from_source[0] == spec.differs_from_source[0]
    assert any("nothing was refitted" in note for note in child.differs_from_source)
    assert child.sources["costs.taker_fee_bps"] == OWNER
    assert "costs" not in child.sources  # stage 1's cost sources are gone
    assert child.sources["signal"] == spec.sources["signal"]  # the parameters keep theirs
    assert source_problems(child) == []


def test_translation_maps_named_instruments_and_drops_what_the_venue_lacks() -> None:
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["BTC", "1000PEPE", "LUNA2", "ETH"]},
                 required_instruments=["BTC", "ETH"])
    result = stage2.translate(spec, CONFIG, venue_names={"BTC", "ETH", "kPEPE", "SOL"})
    assert result.spec.data.instruments == ["BTC", "kPEPE", "ETH"]
    assert result.dropped == ["LUNA2"]
    assert result.spec.required_instruments == ["BTC", "ETH"]
    assert any("LUNA2" in note for note in result.spec.differs_from_source)


def test_a_list_dated_after_the_venue_data_begins_is_judged_from_that_day_here_too() -> None:
    """A causal twin's hand-picked list (`instruments_chosen_at`) knows how earlier days ended:
    the child on Hyperliquid is judged from the list's day, the earlier days are warm-up."""
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["BTC", "ETH"]},
                 instruments_chosen_at=date(2025, 6, 1),
                 sources={**_spec().sources, "instruments_chosen_at": OWNER})
    child = stage2.translate(spec, CONFIG, venue_names={"BTC", "ETH"}).spec
    assert child.judged_from == date(2025, 6, 1) and child.instruments_chosen_at == date(2025, 6, 1)
    assert any("Judged from 2025-06-01" in note for note in child.differs_from_source)
    # Known before the venue's data begins: nothing to cut.
    early = spec.model_copy(update={"instruments_chosen_at": date(2020, 1, 1)})
    assert stage2.translate(early, CONFIG, venue_names={"BTC", "ETH"}).spec.judged_from is None


def test_translation_with_no_instrument_left_makes_no_spec_and_says_why() -> None:
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["LUNA2", "FTT"]})
    result = stage2.translate(spec, CONFIG, venue_names={"BTC"})
    assert result.spec is None and result.dropped == ["LUNA2", "FTT"]
    assert "none of the 2 instruments" in result.problems[0]


def test_naming_instruments_without_the_venue_list_is_a_loud_error() -> None:
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["BTC"]})
    with pytest.raises(ValueError, match="venue_names"):
        stage2.translate(spec, CONFIG)


@pytest.mark.parametrize("name,expected", [
    ("BTC", "BTC"),
    ("1000PEPE", "kPEPE"),          # Binance per-1000 units -> Hyperliquid k
    ("1000000MOG", "kMOG"),
    ("1MBABYDOGE", None),            # not listed there
    ("LUNA2", None),
    ("BTC-SPOT", None),              # no spot column on the venue
])
def test_venue_names_map_unit_prefixes_and_missing_coins(name: str, expected: str | None) -> None:
    assert stage2.venue_name(name, {"BTC", "kPEPE", "kMOG"}) == expected


def test_a_spot_column_is_matched_among_spot_columns() -> None:
    assert stage2.venue_name("BTC-SPOT", {"BTC", "BTC-SPOT"}) == "BTC-SPOT"
    assert stage2.venue_name("ETH-SPOT", {"BTC", "ETH", "BTC-SPOT"}) is None


# ---- spot legs: Binance's `<coin>-SPOT` on Hyperliquid's spot tokens ----------------------

BV2_COINS = ("BTC", "ETH", "SOL", "AVAX")
BV2_INSTRUMENTS = sorted([*BV2_COINS, *(f"{c}-SPOT" for c in BV2_COINS)])
HL_LISTING = {*BV2_COINS, "HYPE", "kPEPE", *(f"{c}-SPOT" for c in BV2_COINS), "HYPE-SPOT"}


def _bv2_shaped(**overrides: object) -> StrategySpec:
    """What the Bv2 specs look like to the translation: explicit spot and perpetual columns."""
    required = [f"{c}-SPOT" for c in BV2_COINS] + list(BV2_COINS)
    data = {"source": "binance", "interval": "1h", "start": date(2023, 4, 28),
            "end": LAST_DAY, "instruments": BV2_INSTRUMENTS}
    return _spec(data=data, required_instruments=required, **overrides)


def _shipped() -> stage2.Stage2Config:
    return stage2.load_config(Path(__file__).resolve().parents[3] / "night" / "stage2.yaml")


def test_the_shipped_config_names_the_four_wrapped_spot_tokens_with_a_source_each() -> None:
    config = _shipped()
    assert {c: leg.token for c, leg in config.spot_legs.items()} == {
        "BTC": "UBTC", "ETH": "UETH", "SOL": "USOL", "AVAX": "UAVAX"}
    assert all(leg.source for leg in config.spot_legs.values())
    assert set(config.no_spot) == {"LINK", "AAVE"} and all(config.no_spot.values())


def test_bv2s_spot_legs_map_to_hyperliquids_spot_columns_and_cite_each_token() -> None:
    config = _shipped()
    result = stage2.translate(_bv2_shaped(), config, venue_names=HL_LISTING)

    assert result.dropped == [] and result.unavailable == []
    child = result.spec
    assert sorted(child.data.instruments) == BV2_INSTRUMENTS
    assert child.data.source == "hyperliquid"
    assert sorted(child.required_instruments) == BV2_INSTRUMENTS
    for coin, token in (("BTC", "UBTC"), ("ETH", "UETH"), ("SOL", "USOL"), ("AVAX", "UAVAX")):
        assert f"{coin}-SPOT is hyperliquid's {token}/USDC spot token" in " ".join(
            child.differs_from_source)
        assert child.sources[f"data.spot_legs.{coin}"] == config.spot_legs[coin].source
    # Every cited file exists (frab's repository is a sibling of q-lab's; a stub stands in).
    assert not [p for p in source_problems(child) if "does not exist" in p
                and "funding-rate-arbitrage" not in p]


def test_a_spot_coin_outside_the_verified_list_is_resolved_by_the_labs_rule_and_says_so() -> None:
    spec = _bv2_shaped().model_copy(update={
        "data": _bv2_shaped().data.model_copy(update={"instruments": ["HYPE", "HYPE-SPOT"]}),
        "required_instruments": ["HYPE", "HYPE-SPOT"]})
    child = stage2.translate(spec, _shipped(), venue_names=HL_LISTING).spec
    assert child.data.instruments == ["HYPE", "HYPE-SPOT"]
    assert "not one of the lab's verified tokens" in " ".join(child.differs_from_source)
    assert "fetch_spot_meta" in child.sources["data.spot_legs.HYPE"]


def test_a_required_spot_leg_hyperliquid_lacks_is_unavailable_with_a_cited_reason() -> None:
    config = _shipped()
    spec = _bv2_shaped().model_copy(update={
        "data": _bv2_shaped().data.model_copy(update={"instruments": [
            "BTC", "BTC-SPOT", "LINK", "LINK-SPOT", "DOGE", "DOGE-SPOT"]}),
        "required_instruments": ["BTC-SPOT", "BTC", "LINK-SPOT", "LINK", "DOGE-SPOT", "DOGE"]})
    listing = {"BTC", "BTC-SPOT", "LINK", "DOGE"}  # perpetuals of LINK and DOGE, no spot

    result = stage2.translate(spec, config, venue_names=listing)

    assert result.dropped == ["LINK-SPOT", "DOGE-SPOT"]
    assert len(result.unavailable) == 2
    link, doge = result.unavailable
    assert link.startswith("LINK-SPOT has no hyperliquid spot leg") and "LINK0" in link
    assert "tokens.py" in link  # frab's blacklist is the cited source
    assert doge.startswith("DOGE-SPOT has no hyperliquid spot leg")
    assert "named DOGE or UDOGE" in doge and "fetch_spot_meta" in doge
    # The leg is not dropped silently from the requirement: the run says it is absent.
    assert "LINK-SPOT" in result.spec.required_instruments


def test_a_required_perpetual_the_venue_lacks_is_unavailable_too() -> None:
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["BTC", "LUNA2"]},
                 required_instruments=["BTC", "LUNA2"])
    result = stage2.translate(spec, CONFIG, venue_names={"BTC"})
    assert result.unavailable == ["LUNA2 is not listed on hyperliquid"]


def test_a_book_missing_a_required_spot_leg_on_the_venue_is_not_evaluable_with_a_trial(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), FAILING)  # a ruleset that would reject anything run
    spec = _spec(data={"source": "binance", "interval": "1d", "start": date(2019, 9, 8),
                       "end": LAST_DAY, "instruments": ["C0", "C1", "C2", "C3", "C4", "C5",
                                                         "LINK", "LINK-SPOT"]},
                 required_instruments=["LINK-SPOT", "LINK"])
    listing = {"C0", "C1", "C2", "C3", "C4", "C5", "LINK"}
    config = dataclasses.replace(CONFIG, no_spot={"LINK": "LINK0 is an EVM bridge (tokens.py)"})

    result = stage2.run_stage2(
        spec, stage1_route="paper", stage1_trial_id=None, noise_trials=1,
        deployable_capital_usd=3000.0, spec_dir=tmp_path / "stage2", config=config,
        venue_names=listing)

    assert result.status == "not-evaluable" and result.route == "not-evaluable"
    assert "LINK-SPOT has no hyperliquid spot leg: LINK0 is an EVM bridge" in result.reason
    assert result.unavailable and result.dropped == ["LINK-SPOT"]
    assert result.combined_route == "needs-forward"  # waits; not a reject
    with db_module.session_scope() as session:
        trial = session.get(Trial, result.trial_id)  # written like every run
        assert trial.status == TrialStatus.NOT_EVALUABLE


def test_the_written_spec_loads_back_the_same(tmp_path) -> None:
    child = stage2.translate(_spec(), CONFIG).spec.model_copy(
        update={"judged_from": date(2025, 1, 30)})
    path = stage2.write_spec(child, PARENT, tmp_path / "out")
    assert path.name == f"{CHILD}.yaml"
    assert path.read_text(encoding="utf-8").startswith(f"# Stage 2 of {PARENT}")
    assert load_spec(path) == child


# ---- the real configuration -------------------------------------------------------------


def test_the_shipped_config_cites_hyperliquids_costs_where_the_checker_finds_them(
        tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[3]
    config = stage2.load_config(root / "night" / "stage2.yaml")
    assert config.enabled and config.venue == "hyperliquid"
    assert (config.taker_fee_bps, config.slippage_bps) == (4.5, 0.9)
    assert set(config.stage1_survives) == {"paper", "shelf", "needs-infrastructure",
                                           "needs-forward"}  # never reject, never undecided
    # frab's repository is a sibling of q-lab's: stand a stub in for it, the slippage is a
    # `derived:` formula that only needs the cited file to exist.
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "SCREENING.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(sources, "FRAB_ROOT", tmp_path)
    child = stage2.translate(_spec(), config).spec
    assert source_problems(child) == []  # 4.5 appears in the cited doc, 0.9 = 4.4 - 3.5


def test_a_config_without_a_key_or_a_cost_source_is_refused(tmp_path) -> None:
    broken = tmp_path / "stage2.yaml"
    broken.write_text("enabled: true\nvenue: hyperliquid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        stage2.load_config(broken)
    broken.write_text(
        "enabled: true\nvenue: hyperliquid\ndata_start: 2025-01-01\n"
        "costs: {taker_fee_bps: 4.5, slippage_bps: 0.9, sources: {costs.taker_fee_bps: x}}\n"
        "stage1_survives: [paper]\nspot_legs: {}\nno_spot: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no source for"):
        stage2.load_config(broken)
    broken.write_text(
        "enabled: true\nvenue: hyperliquid\ndata_start: 2025-01-01\n"
        "costs: {taker_fee_bps: 4.5, slippage_bps: 0.9, sources: {costs.taker_fee_bps: x,"
        " costs.slippage_bps: y}}\nstage1_survives: [paper]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key 'spot_legs'"):
        stage2.load_config(broken)  # the spot mapping is declared, never a silent default


# ---- from which day the market holds the book -------------------------------------------


def test_the_judged_window_starts_when_enough_coins_have_their_history() -> None:
    # Coins 0-3 trade from day 0; coin 4, the fifth the quintile rule needs, from day 40.
    panel = _market(listed_at={4: 40, 5: 40, 6: 40, 7: 40})
    window = stage2.book_window(panel, STAGE1_PARAMS)
    assert window.reason is None and window.needed == 5 and window.block == "long_short_quantile"
    # min_history_days 30: coin 4 is admitted on its 30th priced bar, day 69.
    assert window.judged_from == (panel.prices.index[0] + pd.Timedelta(days=69)).date()
    assert window.judged_bars == DAYS - 69 and window.full_share == 1.0 and window.most == 8


def test_point_in_time_a_coin_not_yet_listed_is_not_counted_before_its_day() -> None:
    panel = _market(n_coins=5, listed_at={4: 100})
    window = stage2.book_window(panel, STAGE1_PARAMS)
    assert window.judged_from == (panel.prices.index[0] + pd.Timedelta(days=129)).date()


def test_too_few_coins_for_the_portfolio_rule_is_a_plain_reason_not_a_fail() -> None:
    window = stage2.book_window(_market(n_coins=3), STAGE1_PARAMS)
    assert window.judged_from is None and window.most == 3 and window.needed == 5
    assert "never had 5 admissible coins at once" in window.reason
    assert "long_short_quantile" in window.reason and "most on any day: 3" in window.reason


def test_a_book_that_exists_only_on_the_last_day_is_too_short_to_measure() -> None:
    window = stage2.book_window(_market(n_coins=5, listed_at={4: DAYS - 30}), STAGE1_PARAMS)
    assert window.judged_from is None and "last day" in window.reason


def test_the_history_gate_is_part_of_the_window() -> None:
    params = {**STAGE1_PARAMS, "min_history_days": DAYS + 50}
    assert "never had" in stage2.book_window(_market(), params).reason


# ---- both stages together ---------------------------------------------------------------


@pytest.mark.parametrize("stage1,stage2_route,expected", [
    ("paper", "paper", "paper"),
    ("paper", "shelf", "shelf"),
    ("shelf", "paper", "paper"),
    ("needs-infrastructure", "paper", "paper"),       # Binance's missing adapter, settled on HL
    ("paper", "needs-forward", "needs-forward"),
    ("paper", "reject", "reject"),                    # either stage rejecting rejects
    ("needs-forward", "reject", "reject"),
    ("needs-forward", "paper", "needs-forward"),      # stage 1 has not passed yet
    ("paper", "not-evaluable", "needs-forward"),      # undecided: the idea waits
    ("paper", "needs-more-data", "needs-forward"),
    ("paper", "error", "needs-forward"),
    ("paper", None, "needs-forward"),
    ("reject", "paper", "reject"),                    # stage 1 reject is final, stage 2 moot
    ("not-evaluable", "paper", "not-evaluable"),
])
def test_an_idea_needs_both_stages_to_survive(stage1: str, stage2_route: str | None,
                                              expected: str) -> None:
    assert stage2.combine(stage1, stage2_route, config=CONFIG).route == expected


def test_only_survivors_of_stage_1_go_on() -> None:
    assert [stage2.survives(r, CONFIG) for r in
            ("paper", "shelf", "needs-infrastructure", "needs-forward", "reject",
             "not-evaluable", "needs-more-data", "error", None)] == [
        True, True, True, True, False, False, False, False, False]


# ---- the step ---------------------------------------------------------------------------


def test_a_stage_2_reject_sends_the_idea_to_the_graveyard_with_stage_2s_data_range(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), FAILING)
    _set_status(PARENT, IdeaStatus.VALIDATED)  # stage 1 passed

    result = _run(tmp_path, "paper")

    assert (result.status, result.route, result.combined_route) == ("evaluated", "reject",
                                                                     "reject")
    assert result.parent_status == "rejected"
    assert _status(PARENT) == IdeaStatus.REJECTED and _status(CHILD) == IdeaStatus.REJECTED
    window_start = result.judged_from
    assert window_start == "2025-01-30" and result.data_end == LAST_DAY.isoformat()
    with db_module.session_scope() as session:
        # The verdicts are the child's, rendered on the venue's data, judged from the first
        # full-book day to the end of its data.
        rows = session.query(Verdict).filter(Verdict.idea_id == CHILD).all()
        assert rows and {v.data_range_start for v in rows} == {date(2025, 1, 30)}
        assert {v.data_range_end for v in rows} == {LAST_DAY}
        assert session.query(Verdict).filter(Verdict.idea_id == PARENT).count() == 0
        move = (session.query(StageTransition).filter(StageTransition.idea_id == PARENT)
                .order_by(StageTransition.id.desc()).first())
        assert move.to_status == IdeaStatus.REJECTED
        assert "stage 2 (hyperliquid) rejected" in move.reason and move.trial_id == result.trial_id
        assert session.get(Idea, CHILD).parent_id == PARENT
        assert stage2_child(session, PARENT).id == CHILD

        # Resurrection only on data the rejection did not have: a LATER run (its own trial,
        # hence trial_id None here) of the parent reads the child's failing verdicts too, so
        # the same data does not revive it and later data does. Without the child's verdicts
        # the date of the rejection would be unknown and nothing could revive it.
        def rerun(run_data_end: date, also=()):
            return apply_route(session, idea_id=PARENT, route=TrialRoute.PAPER, route_reason="r",
                               trial_id=None, rules_version="x", run_data_end=run_data_end,
                               also_ideas=also)

        same = rerun(LAST_DAY, (CHILD,))
        assert same.new_status is None
        assert "rejected on data up to 2025-07-19" in same.reason and "nothing new" in same.reason
        undated = rerun(date(2025, 8, 1))
        assert undated.new_status is None and "undated" in undated.reason
        assert rerun(date(2025, 8, 1), (CHILD,)).new_status == IdeaStatus.VALIDATED


def test_both_stages_passing_leaves_the_idea_validated_for_the_owner(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)
    _set_status(PARENT, IdeaStatus.VALIDATED)

    result = _run(tmp_path, "paper")

    assert (result.route, result.combined_route) == ("paper", "paper")
    assert _status(PARENT) == IdeaStatus.VALIDATED  # never `paper`: the owner's gate
    assert _status(CHILD) == IdeaStatus.VALIDATED
    assert result.dropped == [] and result.needed_coins == 5 and result.most_coins == 8
    assert Path(result.spec_path).is_file()
    with db_module.session_scope() as session:
        child = session.get(Idea, CHILD)
        assert child.notes.startswith(NOTES_PREFIX + PARENT)
        trial = session.get(Trial, result.trial_id)
        assert trial.snapshot_id == SNAPSHOT and trial.status == TrialStatus.OK
        assert trial.params == STAGE1_PARAMS  # the numbers that passed stage 1, recorded as run
        assert trial.metrics["skipped_warmup_days"] == pytest.approx(29.0)


def test_a_stage_2_that_decides_nothing_keeps_the_idea_waiting_not_passed(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)
    _set_status(PARENT, IdeaStatus.VALIDATED)  # stage 1 said paper
    result = _run(tmp_path, "needs-forward")  # stage 1 still waiting for forward data
    assert result.route == "paper" and result.combined_route == "needs-forward"
    assert _status(PARENT) == IdeaStatus.BENCH  # not validated until both have passed


def test_too_few_coins_on_the_venue_is_not_evaluable_with_a_plain_reason_and_a_trial(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(n_coins=3), FAILING)  # a ruleset that would reject anything run
    _set_status(PARENT, IdeaStatus.VALIDATED)

    result = _run(tmp_path, "paper")

    assert result.status == "not-evaluable" and result.route == "not-evaluable"
    assert "never had 5 admissible coins at once" in result.reason
    assert (result.needed_coins, result.most_coins) == (5, 3)
    assert result.combined_route == "needs-forward"  # waits; it is not a fail
    assert _status(PARENT) == IdeaStatus.BENCH and _status(CHILD) == IdeaStatus.CANDIDATE
    with db_module.session_scope() as session:
        trial = session.get(Trial, result.trial_id)  # written like every run
        assert trial.status == TrialStatus.NOT_EVALUABLE
        assert trial.route == TrialRoute.NOT_EVALUABLE and trial.snapshot_id == SNAPSHOT
        assert session.query(Verdict).filter(Verdict.trial_id == trial.id).count() == 0


def test_stage_1_survivors_only_go_on_and_a_reject_is_never_run_on_the_venue(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)
    for route in ("reject", "not-evaluable", "needs-more-data", "error", None):
        result = _run(tmp_path, route)
        assert result.status == "skipped" and "only survivors go on" in result.skipped
    with db_module.session_scope() as session:
        assert session.get(Idea, CHILD) is None  # nothing was created ...
        assert session.query(Trial).count() == 0  # ... and nothing run


def test_a_switched_off_stage_is_skipped_with_that_reason(registry, tmp_path) -> None:
    off = stage2.Stage2Config(**{**CONFIG.__dict__, "enabled": False})
    result = _run(tmp_path, "paper", config=off)
    assert result.status == "skipped" and "switched off" in result.skipped


def test_a_hand_run_measurement_writes_its_trial_but_leaves_the_parents_status(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)
    _set_status(PARENT, IdeaStatus.REJECTED)  # stage 1 rejected it

    result = _run(tmp_path, "reject", measure=True)

    assert result.measurement and result.status == "evaluated" and result.route == "paper"
    assert result.combined_route is None and result.parent_status is None
    assert _status(PARENT) == IdeaStatus.REJECTED  # a measurement is not a route change
    with db_module.session_scope() as session:
        assert session.get(Trial, result.trial_id) is not None
        assert "MEASUREMENT" in session.get(Idea, CHILD).notes


def test_a_crash_of_the_step_is_a_result_not_an_exception_and_the_idea_waits(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)

    def boom(session, spec):
        raise ConnectionError("hyperliquid is unreachable")

    monkeypatch.setattr("qlab.pipeline.evaluate.resolve_panel", boom)
    _set_status(PARENT, IdeaStatus.VALIDATED)

    result = _run(tmp_path, "paper")

    assert result.status == "error" and "ConnectionError" in result.error
    assert result.combined_route == "needs-forward" and _status(PARENT) == IdeaStatus.BENCH
    # Tried again once the venue answers: the same call finishes the job.
    _stand(monkeypatch, _market(), PASSING)
    again = _run(tmp_path, "paper")
    assert again.status == "evaluated" and again.error is None
    assert again.combined_route == "paper" and _status(PARENT) == IdeaStatus.VALIDATED


def test_a_missing_config_is_the_results_error_and_never_breaks_stage_1(
        registry, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no night/stage2.yaml here
    result = stage2.run_stage2(_spec(), stage1_route="paper", stage1_trial_id=3, noise_trials=1,
                               deployable_capital_usd=3000.0, spec_dir=tmp_path / "s")
    assert result.status == "error" and "night/stage2.yaml" in result.error
    assert result.stage1_route == "paper" and _status(PARENT) == IdeaStatus.CANDIDATE


def test_the_stage_2_spec_that_cites_nothing_checkable_is_an_error_not_a_trial(
        registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), PASSING)
    bad = stage2.Stage2Config(**{**CONFIG.__dict__,
                                 "cost_sources": {"costs.taker_fee_bps": "trust me",
                                                  "costs.slippage_bps": OWNER}})
    result = _run(tmp_path, "paper", config=bad)
    assert result.status == "error" and "no checkable source" in result.error
    with db_module.session_scope() as session:
        assert session.query(Trial).count() == 0


# ---- no agent, no tokens ----------------------------------------------------------------


def test_stage_2_calls_no_agent_and_spends_no_tokens(registry, tmp_path, monkeypatch) -> None:
    """Pure code: the step runs to the end with every route to an agent closed."""
    import qlab.agents.generator as generator
    import qlab.budget as budget
    import qlab.budget.agent as budget_agent

    def forbidden(*args, **kwargs):
        raise AssertionError("stage 2 called an agent")

    for module in (budget, budget_agent, generator):
        monkeypatch.setattr(module, "run_agent", forbidden)
    monkeypatch.setenv(budget_agent.CLAUDE_BIN_ENV, "/nonexistent/claude")
    _stand(monkeypatch, _market(), PASSING)

    result = _run(tmp_path, "paper")

    assert result.status == "evaluated"
    with db_module.session_scope() as session:
        from qlab.registry.models import TokenSpend

        assert session.query(TokenSpend).count() == 0
    source = Path(stage2.__file__).read_text(encoding="utf-8")
    assert "run_agent" not in source and "qlab.agents" not in source
    assert "qlab.budget" not in source


# ---- the morning report -----------------------------------------------------------------


def test_the_report_puts_both_stages_in_plain_russian(registry, tmp_path, monkeypatch) -> None:
    _stand(monkeypatch, _market(), FAILING)
    _set_status(PARENT, IdeaStatus.VALIDATED)
    result = _run(tmp_path, "paper")
    text = "\n".join(stage2.report_lines({"route": "paper"}, stage2_dict(result)))
    assert "второй этап, Hyperliquid (тот же механизм и те же числа" in text
    assert "токены не тратятся" in text
    assert "отвергнута" in text
    assert "считается с 2025-01-30 — первого дня, когда на Hyperliquid хватает монет" in text
    assert "итог по двум этапам: отвергнута на втором этапе" in text
    assert "воскресить можно только на данных, которых тогда не было" in text


def stage2_dict(result: stage2.Stage2Result) -> dict:
    from dataclasses import asdict

    return asdict(result)


def test_the_report_explains_a_skip_a_wait_and_a_measurement_in_words(
        registry, tmp_path, monkeypatch) -> None:
    skipped = stage2.report_lines({"route": "reject"}, stage2_dict(_run(tmp_path, "reject")))
    assert "не проводился: на первом этапе исход «отвергнута»" in skipped[0]

    _stand(monkeypatch, _market(n_coins=3), PASSING)
    _set_status(PARENT, IdeaStatus.VALIDATED)
    waiting = "\n".join(stage2.report_lines({"route": "paper"},
                                            stage2_dict(_run(tmp_path, "paper"))))
    assert "проверить нечем — на Hyperliquid ни в один день не набиралось 5 монет сразу" in waiting
    assert "(больше всего было 3)" in waiting and "Это не отказ: идея просто ждёт" in waiting

    measured = "\n".join(stage2.report_lines(
        {"route": "reject"}, stage2_dict(_run(tmp_path, "reject", measure=True))))
    assert "ЗАМЕР на Hyperliquid" in measured and "статус идеи не меняет" in measured
    assert "итог по двум этапам" not in measured
    assert stage2.report_lines({"route": "paper"}, None) == []  # a run from before stage 2


def test_child_id_is_the_parents_with_the_venue_suffix() -> None:
    assert child_id(PARENT) == CHILD
    assert _spec().idea_id == PARENT


# ---- confirmation by stage 1's history (rules 2026-10-27.1) -----------------------------

NOISE_RULE = Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="noise_return_percentile",
                  comparator=Comparator.GE, threshold=0.99, fatal=True,
                  fit_period=FitPeriodUse.INFORMATIONAL, on_short_forward=ShortForwardUse.WAIT)
CONFIRMING = RuleSet(
    version="2026-10-27.1", rules=[NOISE_RULE, CAPITAL_FIT_GENEROUS],
    holdout_share=HoldoutShare(share=1 / 3, rationale="t"),
    noise_power=NoisePower(good_return=0.07, min_power=0.80, rationale="t"),
    stage2_confirmation=Stage2Confirmation(alert_percentile_from="night/decay.yaml",
                                           rationale="t"))


def _trending_market() -> MarketPanel:
    """Coins whose drifts differ a lot, so the momentum mechanism earns something."""
    index = pd.date_range(pd.Timestamp(FIRST_DAY, tz="UTC"), periods=DAYS, freq="1D")
    rng = np.random.default_rng(5)
    drifts = np.linspace(-0.004, 0.004, 8)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(drifts, 0.01, (DAYS, 8)), axis=0),
                          index=index, columns=[f"C{i}" for i in range(8)])
    tradeable = prices.notna()
    funding = pd.DataFrame(0.0, index=index, columns=prices.columns)
    volume = pd.DataFrame(1e6, index=index, columns=prices.columns)
    return MarketPanel(
        snapshot_id=SNAPSHOT, prices=prices, funding=funding, tradeable=tradeable,
        volume=volume,
        meta={"venue": "hyperliquid", "interval": "1d", "universe_complete": True,
              "range_start": index[0].isoformat(), "range_end": index[-1].isoformat()})


def _stand_without_power(monkeypatch, history: confirm.Stage1History | Exception | None,
                         *, noise_power: float = 0.05) -> list:
    """Stage 2 as on Hyperliquid's short window: the candidate loses the noise comparison
    (93.5% against the 99% bar) and the bar has `noise_power`; stage 1's history is `history`.
    Returns the list the loader's calls are appended to."""
    _stand(monkeypatch, _trending_market(), CONFIRMING)
    calls: list = []

    def bar(spec, *, session, ruleset, deployable_capital_usd, n_trials, review_route=None):
        return types.SimpleNamespace(candidate=evaluate_spec(
            spec, session=session, ruleset=ruleset, deployable_capital_usd=deployable_capital_usd,
            check_lookahead=False, review_route=review_route,
            extra_metrics={"noise_return_percentile": 0.935, "noise_power": noise_power}))

    monkeypatch.setattr("qlab.calibration.shape_aware.evaluate_spec_with_shape_aware_bar", bar)

    def loader(session, spec, trial_id):
        calls.append((spec.idea_id, trial_id))
        if isinstance(history, Exception):
            raise history
        return history

    monkeypatch.setattr(confirm, "load_stage1_history", loader)
    return calls


def _history(drift: float = 0.002, noise: float = 0.03, days: int = 900,
             power: float | None = 0.95) -> confirm.Stage1History:
    index = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    returns = np.random.default_rng(2).normal(drift, noise, days)
    return confirm.Stage1History(41, pd.Series(returns, index=index), power)


def _confirm_run(tmp_path, route: str | None = "paper") -> stage2.Stage2Result:
    _set_status(PARENT, IdeaStatus.VALIDATED)
    return stage2.run_stage2(_spec(), stage1_route=route, stage1_trial_id=41, noise_trials=1,
                             deployable_capital_usd=3000.0, spec_dir=tmp_path / "stage2",
                             config=CONFIG)


def test_without_power_a_consistent_result_confirms_stage_1_and_the_stand_records_it(
        registry, tmp_path, monkeypatch) -> None:
    calls = _stand_without_power(monkeypatch, _history())

    result = _confirm_run(tmp_path)

    assert calls == [(PARENT, 41)]  # stage 1's own idea and its trial id
    assert (result.route, result.combined_route) == ("paper", "paper")
    assert _status(PARENT) == IdeaStatus.VALIDATED and _status(CHILD) == IdeaStatus.VALIDATED
    found = result.confirmation
    assert found["status"] == "confirmed" and found["rules_version"] == "2026-10-27.1"
    assert found["stage1_trial_id"] == 41 and found["alert_percentile"] == 7.0
    assert found["segments"] > 100 and found["independent_segments"] >= 1
    assert found["percentile"] > 7.0 and found["total_return"] > 0
    with db_module.session_scope() as session:
        trial = session.get(Trial, result.trial_id)
        assert trial.route == TrialRoute.PAPER  # the trial carries the route that stands
        assert "consistency check, not a second proof" in trial.route_reason
        assert trial.metrics["stage1_segment_percentile"] == pytest.approx(found["percentile"])
        assert trial.metrics["stage1_segments"] == found["segments"]
        assert trial.metrics["stage1_trial_id"] == 41
        verdicts = {v.rule_id: v for v in session.query(Verdict).filter(
            Verdict.trial_id == trial.id, Verdict.rule_id.in_(
                ["stage1_consistency", "stage2_total_return_positive"])).all()}
        assert set(verdicts) == {"stage1_consistency", "stage2_total_return_positive"}
        assert all(v.passed and v.rules_version == "2026-10-27.1" for v in verdicts.values())
        assert verdicts["stage1_consistency"].threshold == 7.0
        assert verdicts["stage1_consistency"].data_range_end == LAST_DAY
        assert "confirmation by stage 1's history" in verdicts["stage1_consistency"].note
        noise = session.query(Verdict).filter(
            Verdict.trial_id == trial.id, Verdict.rule_id == "shape_aware_edge",
            Verdict.note.like("forward test%")).one()
        assert noise.passed is False  # the noise comparison's own verdict is still recorded


def test_without_power_a_result_unlike_stage_1s_history_is_rejected_with_stage_2s_range(
        registry, tmp_path, monkeypatch) -> None:
    _stand_without_power(monkeypatch, _history(drift=0.05, noise=0.002))  # a far better history

    result = _confirm_run(tmp_path)

    assert result.route == "reject" and result.combined_route == "reject"
    assert result.confirmation["status"] == "inconsistent"
    assert result.confirmation["percentile"] <= result.confirmation["alert_percentile"]
    assert _status(PARENT) == IdeaStatus.REJECTED and _status(CHILD) == IdeaStatus.REJECTED
    with db_module.session_scope() as session:
        failed = session.query(Verdict).filter(
            Verdict.trial_id == result.trial_id, Verdict.passed.is_(False),
            Verdict.rule_id == "stage1_consistency").one()
        assert failed.data_range_end == LAST_DAY  # the graveyard's data range is stage 2's
    text = "\n".join(stage2.report_lines({"route": "paper"}, stage2_dict(result)))
    assert "хуже нынешнего было 0 из 100 таких же отрезков на Binance" in text
    assert "порог владельца — нижние 7" in text and "на себя не похоже; отказ" in text


def test_a_powered_noise_comparison_still_decides_as_before(
        registry, tmp_path, monkeypatch) -> None:
    calls = _stand_without_power(monkeypatch, _history(), noise_power=0.90)

    result = _confirm_run(tmp_path)

    assert calls == [] and result.confirmation is None  # stage 1's history is not even read
    assert result.route == "reject" and result.combined_route == "reject"


def test_stage_1_still_too_early_never_turns_into_a_pass(registry, tmp_path, monkeypatch) -> None:
    calls = _stand_without_power(monkeypatch, _history())

    result = _confirm_run(tmp_path, "needs-forward")

    assert calls == [] and result.confirmation is None
    assert result.route == "needs-forward" and result.combined_route == "needs-forward"
    assert _status(PARENT) == IdeaStatus.BENCH
    # and were stage 2 to pass anyway, combine keeps the idea waiting on stage 1
    assert stage2.combine("needs-forward", "paper", config=CONFIG).route == "needs-forward"


def test_without_stage_1_history_the_old_route_stands_and_the_reason_says_why(
        registry, tmp_path, monkeypatch) -> None:
    _stand_without_power(monkeypatch, confirm.Stage1HistoryMissing(
        "stage 1's data snapshot snap-9 cannot be read: gone"))

    result = _confirm_run(tmp_path)

    assert result.route == "needs-forward" and result.combined_route == "needs-forward"
    assert _status(PARENT) == IdeaStatus.BENCH
    assert result.confirmation["status"] == "not-applied" and result.confirmation["route"] is None
    assert ("consistency check with stage 1 not applied: stage 1's data snapshot snap-9 "
            "cannot be read: gone") in result.reason
    assert result.reason.startswith("too early to tell:")  # today's reason, kept
    text = "\n".join(stage2.report_lines({"route": "paper"}, stage2_dict(result)))
    assert "сверка с длинной историей не проведена" in text and "идея ждёт" in text
    assert "сравнение со случайными книгами на таком коротком отрезке" in text


def test_a_hand_run_measurement_is_never_confirmed(registry, tmp_path, monkeypatch) -> None:
    calls = _stand_without_power(monkeypatch, _history())
    _set_status(PARENT, IdeaStatus.REJECTED)

    result = stage2.run_stage2(_spec(), stage1_route="reject", stage1_trial_id=41,
                               noise_trials=1, deployable_capital_usd=3000.0,
                               spec_dir=tmp_path / "stage2", config=CONFIG, measure=True)

    assert calls == [] and result.confirmation is None and result.route == "needs-forward"


def test_the_report_of_a_confirmed_idea_says_what_was_compared_in_plain_words(
        registry, tmp_path, monkeypatch) -> None:
    _stand_without_power(monkeypatch, _history())
    text = "\n".join(stage2.report_lines({"route": "paper"},
                                         stage2_dict(_confirm_run(tmp_path))))
    assert "сверка с длинной историей: хуже нынешнего было" in text
    assert "таких же отрезков на Binance (порог владельца — нижние 7)" in text
    assert "из них не пересекающихся" in text and "прогоном первого этапа 41" in text
    assert "правила 2026-10-27.1" in text
    assert "решение владельца 2026-10-27" in text
    assert "а не второе доказательство" in text
    assert "прошла оба этапа" in text and "второй подтвердил согласие с ним" in text


# ---- the second stage's id belongs to another idea ---------------------------------------


def test_a_second_stage_never_writes_into_an_idea_that_already_has_its_id(
        tmp_path, registry) -> None:
    """bv2-research-shared-pool-hl is a researcher's experiment, and also the id the second
    stage of bv2-research-shared-pool would take: its trials and status are not the second
    stage's to write (found 2026-10-27, when one trial and a spec version landed there)."""
    child = f"{PARENT}-hl"
    with registry.session_scope() as session:
        repo.upsert_idea(session, id=child, title="another experiment",
                         source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                         profile=Profile.OTHER, notes="Declared by the researcher (bv2).")
        before = session.get(Idea, child).status
    result = _run(tmp_path)
    assert result.skipped and "belongs to another idea" in result.skipped
    assert result.trial_id is None and result.status == "skipped"
    assert not (tmp_path / "stage2").exists()  # no spec written either
    assert _status(child) == before and _status(PARENT) != IdeaStatus.REJECTED
    with registry.session_scope() as session:
        assert session.get(Idea, child).parent_id is None  # not adopted as a second stage
        from qlab.registry.stages import child_id_taken

        assert child_id_taken(session, PARENT)
    lines = stage2.report_lines({"route": "paper"}, dataclasses.asdict(result))
    assert "уже занято другой идеей" in lines[0]
