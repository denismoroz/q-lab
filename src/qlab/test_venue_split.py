"""The venue split (docs/VENUE_SPLIT.md, owner 2026-10-30): Binance and Hyperliquid are two venues,
each judged on its own data, coins, costs and minimum leg. Pure readings of the cost table, the
venue files, the rules and the spec rewrite; no registry, no data."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from qlab import costs, venue_split
from qlab.pipeline import confirm
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.rules.loader import load, load_latest
from qlab.venues.config import leg_floors, load_venue

ROOT = Path(__file__).resolve().parents[2]


def test_the_binance_row_has_a_two_leg_number_with_a_checkable_source() -> None:
    row = costs.venue_costs("binance")
    assert (row.taker_fee_bps, row.slippage_bps, row.two_leg_taker_fee_bps) == (5.0, 0.9, 7.5)
    assert costs.table_citation_problems() == []  # the (5 + 10) / 2 formula gives 7.5
    assert costs.venue_costs("hyperliquid").two_leg_taker_fee_bps is None


def test_a_two_leg_book_on_binance_takes_the_mean_of_the_perpetual_and_the_spot_fee() -> None:
    two, one = SimpleNamespace(simultaneous_legs=2), SimpleNamespace(simultaneous_legs=1)
    row = costs.venue_costs("binance")
    assert costs.expected_taker_fee_bps(two, row) == 7.5
    assert costs.expected_taker_fee_bps(one, row) == 5.0
    assert venue_split.numbers(2).taker_fee_bps == 7.5
    assert venue_split.numbers(1).taker_fee_bps == 5.0


def test_binances_minimum_leg_is_the_published_one_per_contract() -> None:
    """The venue file's per-contract exceptions are exactly the rows of the exchange table that are
    not the ordinary 5 (seed/venue_facts/2026-10-30-binance_um_min_notional.csv, read from
    exchangeInfo on 2026-10-30)."""
    table = ROOT / "seed" / "venue_facts" / "2026-10-30-binance_um_min_notional.csv"
    with table.open(encoding="utf-8") as fh:
        rows = {r["baseAsset"]: float(r["MIN_NOTIONAL_usd"]) for r in csv.DictReader(fh)}
    venue = load_venue("binance")
    ordinary = venue.min_leg_notional_usd
    assert ordinary == 5.0 and sum(v == ordinary for v in rows.values()) == 516
    assert {k: v for k, v in rows.items() if v != ordinary} == venue.min_leg_notional_by_instrument
    assert venue.min_leg_notional_by_instrument["BTC"] == 50.0  # the number the file used to carry
    assert venue.execution_adapter is False  # a fact about our code, not about access


def test_leg_floors_give_a_contract_its_own_number_and_a_spot_leg_the_spot_one() -> None:
    floors = leg_floors(load_venue("binance"), ["BTC", "ETH", "SOL", "BTC-SPOT", "SOL-SPOT"])
    assert floors == {"BTC": 50.0, "ETH": 20.0, "SOL": 5.0, "BTC-SPOT": 5.0, "SOL-SPOT": 5.0}
    assert leg_floors(load_venue("hyperliquid"), ["BTC"]) is None  # one number for the venue
    assert leg_floors(None, ["BTC"]) is None


def _spec(**over) -> StrategySpec:
    base = load_spec(ROOT / "specs" / "research" / "xsmom" / "xsmom-research-liq1m-causal.yaml")
    return base.model_copy(update=over) if over else base


def test_the_open_binance_spec_files_are_judged_for_binance() -> None:
    spec = _spec()
    assert spec.execution_venue is None and costs.order_venue(spec) == "binance"
    assert (spec.costs.taker_fee_bps, spec.costs.slippage_bps) == (5.0, 0.9)
    assert spec.min_leg_notional == 5.0 and costs.DEVIATION_KEY not in spec.sources
    assert spec.params["score_costs"] == {"taker_fee_bps": 5.0, "slippage_bps": 0.9}


def test_a_spec_made_the_old_way_becomes_the_binance_verdict_of_its_idea() -> None:
    from qlab.pipeline.spec import SpecCosts

    old = _spec(execution_venue="hyperliquid", costs=SpecCosts(taker_fee_bps=3.5, slippage_bps=0.9),
                min_leg_notional=12.0)
    old.params["score_costs"] = {"taker_fee_bps": 3.5, "slippage_bps": 0.9}
    new = venue_split.binance_version(old)
    assert new.execution_venue is None
    assert (new.costs.taker_fee_bps, new.min_leg_notional) == (5.0, 5.0)
    assert new.params["score_costs"]["taker_fee_bps"] == 5.0
    assert new.sources["costs.taker_fee_bps"] == venue_split.numbers(1).taker_source
    assert "venue split" in new.differs_from_source[-1].lower()
    assert old.params["score_costs"]["taker_fee_bps"] == 3.5  # the original is untouched


def test_the_text_of_a_spec_file_keeps_its_comments_and_changes_its_numbers() -> None:
    text = (
        "# a hand-written note about this spec\n"
        "idea_id: x\nsimultaneous_legs: 2\ncosts:\n  taker_fee_bps: 5.25\n  slippage_bps: 0.9\n"
        "min_leg_notional: 10.0\n# the venue note\nexecution_venue: hyperliquid\n"
        "noise_universe: panel\nsources:\n  costs.taker_fee_bps: old\n"
        "  costs.differs_from_table: because\n")
    out = venue_split.binance_text(text)
    assert out.startswith("# a hand-written note about this spec\n")
    assert "# the venue note" in out and "execution_venue" not in out
    assert "taker_fee_bps: 7.5" in out and "min_leg_notional: 5\n" in out
    assert "differs_from_table" not in out and "differs_from_source:" in out


def test_the_latest_rules_split_the_venues_and_older_rules_do_not() -> None:
    latest = load_latest()
    assert latest.venue_split is not None and latest.venue_split.overall == "best_of_venues"
    assert latest.venue_split.shared_coins_history
    assert load("2026-10-29.3").venue_split is None  # older verdicts stay reproducible
    # no threshold moved: the deciding rows are those of 2026-10-29.3
    from qlab.restamp import decisive_part

    was, now = decisive_part(load("2026-10-29.3")), decisive_part(latest)
    now.pop("venue_split")
    was.pop("venue_split")
    assert was == now


def _panel(columns: list[str]):
    index = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame(1.0, index=index, columns=columns)

    class Panel:
        prices = frame
        calls: list = []

        def restrict(self, keep):
            self.calls.append(list(keep))
            return SimpleNamespace(prices=frame[list(keep)], restricted=list(keep))

    return Panel()


def test_stage_1s_history_is_taken_on_the_coins_the_second_venue_also_lists(monkeypatch) -> None:
    seen = {}

    def fake_run(spec, panel):
        seen["columns"] = list(panel.prices.columns)
        return SimpleNamespace(net_return=pd.Series([0.0, 0.01, -0.01]))

    monkeypatch.setattr(confirm.decay, "run_on_panel", fake_run)
    panel = _panel(["BTC", "1000PEPE", "LUNA", "ETH"])
    run, shared, total, why = confirm._run_on_shared_coins(
        None, panel, ["BTC", "ETH", "kPEPE"])  # Hyperliquid names the 1000PEPE contract kPEPE
    assert run is not None and why is None and (shared, total) == (3, 4)
    assert seen["columns"] == ["BTC", "1000PEPE", "ETH"]  # LUNA was never listed there


def test_when_the_shared_coins_cannot_be_used_the_whole_history_stands_and_says_why(
        monkeypatch) -> None:
    panel = _panel(["BTC", "LUNA"])
    run, shared, total, why = confirm._run_on_shared_coins(None, panel, ["DOGE"])
    assert run is None and (shared, total) == (0, 2) and "lists none" in why

    def broken(spec, panel):
        raise ValueError("needs six coins")

    monkeypatch.setattr(confirm.decay, "run_on_panel", broken)
    run, shared, total, why = confirm._run_on_shared_coins(None, panel, ["BTC"])
    assert run is None and shared == 1 and "could not be run" in why and "needs six coins" in why

    monkeypatch.setattr(confirm.decay, "run_on_panel",
                        lambda spec, panel: SimpleNamespace(net_return=pd.Series([0.0, 0.0])))
    run, shared, total, why = confirm._run_on_shared_coins(None, panel, ["BTC"])
    assert run is None and "held nothing" in why


def test_a_binance_pass_routes_needs_infrastructure_by_the_venues_own_facts() -> None:
    """The route of a Binance run that passed every strategy rule is `needs-infrastructure`
    because the venue file says the lab has no adapter; with `execution_venue` omitted nothing
    borrows Hyperliquid's."""
    from qlab.pipeline.evaluate import proxy_execution_venue
    from qlab.venues.derive import derive_venue_metrics

    spec = _spec()
    assert proxy_execution_venue(spec) is None
    facts = derive_venue_metrics(load_venue("binance"), snapshot_source="binance",
                                 venue_id="binance", simultaneous_legs=spec.simultaneous_legs)
    assert facts["venue_supported"] == 0.0
    own = derive_venue_metrics(load_venue("hyperliquid"), snapshot_source="hyperliquid",
                               venue_id="hyperliquid", simultaneous_legs=spec.simultaneous_legs)
    assert own["venue_supported"] == 1.0


@pytest.mark.parametrize("name", ["bv2-binance", "xsmom-research-liq1m-causal"])
def test_no_open_binance_idea_is_left_with_hyperliquids_numbers(name) -> None:
    found = [p for p in (ROOT / "specs").rglob(f"{name}.yaml")]
    assert len(found) == 1
    spec = load_spec(found[0])
    assert spec.data.source == "binance" and spec.execution_venue is None
    assert spec.min_leg_notional == 5.0


def test_a_missing_adapter_does_not_turn_a_wait_into_a_reject() -> None:
    """A Binance run fails `venue_supported` (a fact about our code); counted as a failure of the
    strategy it made every Binance run whose noise bar cannot yet tell a reject instead of a wait
    (found 2026-10-30 on xsmom-research-nobull-causal). From rules 2026-10-30.1 it is left out."""
    from qlab.pipeline.evaluate import RuleKind, _failed_for_waiting

    rows = [SimpleNamespace(rule_id="shape_aware_edge", passed=False),
            SimpleNamespace(rule_id="venue_supported", passed=False),
            SimpleNamespace(rule_id="net_edge_positive", passed=True)]
    by_id = {"shape_aware_edge": SimpleNamespace(kind=RuleKind.STRATEGY),
             "venue_supported": SimpleNamespace(kind=RuleKind.INFRASTRUCTURE)}
    new, old = load_latest(), load("2026-10-29.3")
    assert [r.rule_id for r in _failed_for_waiting(new, by_id, rows)] == ["shape_aware_edge"]
    assert [r.rule_id for r in _failed_for_waiting(old, by_id, rows)] == [
        "shape_aware_edge", "venue_supported"]  # older verdicts stay reproducible
