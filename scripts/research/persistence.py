"""Choose the switching penalty for trend on the development period, as the
statistical jump model does (docs/TASKS.md T39; owner, 2026-10-04: «делай
устойчивость режима для trend»).

For each penalty of the grid declared before any run (0, 0.5, 1, 2, 4, 8 --
in units of log-probability; 0 is the plain most probable regime) the
persistent regime (`qlab.regime_detect.persistent` over the stored level-1
probabilities) is used two ways:

- gate: trend holds cash on persistent-flat days (trend-binance-cap-gated's
  rule with this regime);
- two levels: trend's level 2 (`qlab.meta_label`) with the persistent regime's
  flags in place of the probabilities.

Each variant is trend's own weights masked by its decision, run through the
harness with the spec's costs -- so leaving and re-entering the market costs
what it costs. Development only (2020-08 .. 2023-12) is printed; the chosen
variant is scored on 2024-01 on once with `--holdout VARIANT PENALTY`.
"""

from __future__ import annotations

import sys
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

_sys.path.insert(0, str(Path(__file__).parent))
from _record import log  # noqa: E402

from qlab import meta_label as ml
from qlab import regime_detect as rd
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics
from qlab.harness.run import run_backtest
from qlab.pipeline.evaluate import _slice_rows, resolve_panel, resolve_strategy
from qlab.pipeline.spec import load_spec
from qlab.regimes import market_closes
from qlab.registry.db import get_sessionmaker

GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
# Development result (2026-10-04), read before the holdout was scored: the gate
# improves steadily with the penalty, best at 8 (+11.3% a year, Sharpe 0.91,
# against +5.8% / 0.47 ungated; 12 regime changes a year instead of 57); the
# two-level variant with persistent flags is worse at every penalty. Chosen:
# gate, penalty 8 -- the edge of the declared grid, which is said, not hidden.
CHOICE = ("gate", 8.0)
DEV = (pd.Timestamp("2020-08-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"))


def main() -> None:
    spec = load_spec(Path("specs/trend-binance-cap.yaml"))
    session = get_sessionmaker()()
    try:
        panel = resolve_panel(session, spec)
    finally:
        session.rollback()
        session.close()
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    net = run_backtest(panel, weights, costs, panel.funding).net_return
    returns = net.set_axis(net.index + pd.Timedelta(days=1))  # by deciding close
    level1 = rd.load_predictions("btc-returns-logistic")
    btc = market_closes()

    def mask(decision_by_close: pd.Series) -> pd.Series:
        """Decision per panel row: the latest stamped by a close at or
        before the row (RegimeGate's convention)."""
        known = decision_by_close.dropna().astype(float)
        return known.reindex(panel.prices.index, method="ffill").fillna(0.0)

    def measure(on: pd.Series, window) -> dict[str, float]:
        lo = int(panel.prices.index.searchsorted(window[0]))
        hi = min(int(panel.prices.index.searchsorted(window[1])), len(panel.prices) - 1)
        part = _slice_rows(panel, lo, hi)
        w = weights.mul(on, axis=0).iloc[lo:hi + 1]
        m = compute_metrics(run_backtest(part, w, costs, part.funding))
        return {"ann": m["ann_return_net"], "sharpe": m["sharpe_net"], "dd": m["max_dd"],
                "in_market": float((on.iloc[lo:hi + 1] > 0).mean())}

    def variant(kind: str, penalty: float) -> pd.Series:
        state = rd.persistent(level1, penalty)
        if kind == "gate":
            return mask((state != "flat").where(state.notna()))
        feats = ml.features(level1, btc, returns, state=state)
        dec = ml.walk_forward(feats, returns, level1.index[0])
        return mask(dec["trade"])

    def switches(penalty: float) -> float:
        s = rd.persistent(level1, penalty).dropna()
        s = s[(s.index >= DEV[0]) & (s.index < DEV[1])]
        return float((s != s.shift()).sum() / (len(s) / 365.25))

    if "--holdout" in sys.argv:
        k = sys.argv.index("--holdout")
        kind, penalty = sys.argv[k + 1], float(sys.argv[k + 2])
        r = measure(variant(kind, penalty), HOLDOUT)
        log("trend-gate-persistence", f"{kind}, penalty {penalty}", HOLDOUT,
            {"ann_return": r["ann"], "sharpe": r["sharpe"], "max_dd": r["dd"],
             "in_market": r["in_market"]}, "scripts/research/persistence.py",
            chosen=(kind, penalty) == CHOICE)
        print(f"holdout {kind} penalty {penalty}: "
              + " ".join(f"{a} {b:.3f}" for a, b in r.items()))
        return
    print(f"trend ungated: {measure(pd.Series(1.0, index=panel.prices.index), DEV)}")
    for penalty in GRID:
        line = f"penalty {penalty:>3} ({switches(penalty):.0f} changes a year)"
        for kind in ("gate", "two levels"):
            r = measure(variant(kind, penalty), DEV)
            log("trend-gate-persistence", f"{kind}, penalty {penalty}", DEV,
                {"ann_return": r["ann"], "sharpe": r["sharpe"], "max_dd": r["dd"],
                 "changes_per_year": switches(penalty)}, "scripts/research/persistence.py")
            line += (f" | {kind}: ann {r['ann']:+.3f} sharpe {r['sharpe']:.2f} "
                     f"dd {r['dd']:.3f} in {r['in_market']:.0%}")
        print(line, flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
