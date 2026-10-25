"""Sensitivity runs of FRAB-with-the-spot-on-Binance (docs/FRAB_BINANCE_SPOT.md):
slot counts, slot sizes and the transfer delay. Every cell is a TRIAL in the
registry (CLAUDE.md: a trial is written for every run, so the deflation counts
them), under the measurement idea `frab-binance-spot-grid`, with the idea's status
left alone. A cell is a measurement of one strategy at another size, not a
candidate and not a verdict; the cells' params are not cited to a source on
purpose (a grid value is a question, not a choice), which is why the source check
is off for them and only for them.

Usage:
    uv run python scripts/frab_binance_spot_grid.py            # the whole grid
    uv run python scripts/frab_binance_spot_grid.py --list     # the cells, no runs

The slot's footprint (budget / slots) is the live one, $22.5, unless the cell says
otherwise; a bigger footprint is a bigger book, with the stand's flat 0.9 bp
slippage (unmeasured at size) -- read those rows as a floor on the costs.
"""

from __future__ import annotations

import sys

from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.spec import load_spec
from qlab.registry import repo
from qlab.registry.db import session_scope
from qlab.registry.models import AssetClass, Idea, IdeaStatus, Profile, SourceType
from qlab.rules.loader import load_latest

BINANCE = ("specs/frab-binance-spot.yaml", "frab-binance-spot-grid", "frab-binance-spot")
LIVE_OWN = ("specs/research/frab/frab-live-own-spot.yaml", "frab-live-own-spot-grid",
            "perp-funding-premium")
LIVE_FOOTPRINT = 22.5

#: (which spec, slots, footprint in dollars, extra params)
CELLS: list[tuple[tuple[str, str, str], int, float, dict[str, object]]] = [
    # more slots at the live slot size -- the owner's point, more coins
    *[(BINANCE, n, LIVE_FOOTPRINT, {}) for n in (6, 8, 12, 20)],
    # bigger books: the flat transfer fees shrink against the position
    *[(BINANCE, n, f, {}) for n in (4, 8, 12, 20) for f in (250.0, 1000.0)],
    # the other end of the margin policy, and a slower transfer
    (BINANCE, 4, LIVE_FOOTPRINT, {"margin_return": "on_close"}),
    (BINANCE, 4, 250.0, {"margin_return": "on_close"}),
    (BINANCE, 4, LIVE_FOOTPRINT, {"transfer_delay_hours": 6}),
    (BINANCE, 4, LIVE_FOOTPRINT, {"transfer_delay_hours": 24}),
    (BINANCE, 4, 1000.0, {"transfer_delay_hours": 24}),
    # FRAB live (spot on Hyperliquid, six coins) at the same sizes, as the comparator
    *[(LIVE_OWN, n, f, {}) for n in (4, 6) for f in (LIVE_FOOTPRINT, 250.0, 1000.0)],
]


def cell_spec(which: tuple[str, str, str], slots: int, footprint: float,
              extra: dict[str, object]):
    path, grid_idea, _driver = which
    spec = load_spec(path)
    params = {**spec.params, "concurrency_cap": slots, "budget_cap_usdc": slots * footprint,
              **extra}
    note = ", ".join(f"{k}={v}" for k, v in extra.items())
    return spec.model_copy(update={
        "idea_id": grid_idea,
        "title": f"Измерение {spec.idea_id}: {slots} мест по {footprint:g} долларов"
                 + (f", {note}" if note else ""),
        "params": params})


def main(argv: list[str]) -> None:
    if "--list" in argv:
        for cell in CELLS:
            print(cell)
        return
    ruleset = load_latest()
    for _path, grid_idea, driver in {c[0] for c in CELLS}:
        with session_scope() as session:
            if session.get(Idea, grid_idea) is None:
                repo.upsert_idea(
                    session, id=grid_idea, title=f"Измерения {grid_idea[:-5]} при других размерах",
                    source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                    profile=Profile.CARRY, driver_id=driver,
                    notes="Sensitivity cells (slots, slot size, margin policy, transfer delay): "
                          "measurements, not candidates. scripts/frab_binance_spot_grid.py, "
                          "docs/FRAB_BINANCE_SPOT.md.")
                # Never a candidate: retired at birth, so it takes no place among the
                # lab's active strategies or variants (night/limits.yaml).
                repo.set_status(
                    session, idea_id=grid_idea, new_status=IdeaStatus.RETIRED,
                    reason="measurement container for sensitivity cells, never a candidate "
                           "(task of the owner's request of 2026-10-25, docs/FRAB_BINANCE_SPOT.md)")
    for which, slots, footprint, extra in CELLS:
        spec = cell_spec(which, slots, footprint, extra)
        with session_scope() as session:
            result = evaluate_spec(
                spec, session=session, ruleset=ruleset,
                deployable_capital_usd=slots * footprint, update_idea_status=False,
                check_lookahead=False, check_sources=False)
        m = result.metrics or {}
        print(f"{which[1]:<26} slots {slots:>2} x ${footprint:>6g} {extra}  trial "
              f"{result.trial_id}  {result.routing.route:<14} "
              f"ann {m.get('ann_return_net', float('nan')):+.3f}  "
              f"sharpe {m.get('sharpe_net', float('nan')):.2f}  "
              f"maxdd {m.get('max_dd', float('nan')):+.3f}  err {result.error}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
