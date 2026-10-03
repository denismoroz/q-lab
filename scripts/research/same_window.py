"""Bv2 variants judged on one common window (scripts/research): each spec's
weights are computed on its full panel, as the stand does, and every run is
measured from the latest first judged bar among them. Reads data, writes
nothing to the registry (session rolled back)."""
import sys
from pathlib import Path

from qlab import regimes
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics, periods_per_year
from qlab.harness.run import run_backtest
from qlab.pipeline.evaluate import (
    _slice_rows,
    complete_book_window,
    resolve_panel,
    resolve_strategy,
)
from qlab.pipeline.spec import load_spec
from qlab.registry.db import get_sessionmaker

session = get_sessionmaker()()
runs = {}
try:
    for path in sys.argv[1:]:
        spec = load_spec(Path(path))
        panel = resolve_panel(session, spec)
        weights = resolve_strategy(spec.code_ref).run(panel, spec.params)
        weights = getattr(weights, "weights", weights)
        cov = complete_book_window(panel, spec.required_instruments)
        runs[path] = (spec, panel, weights, cov.window)
finally:
    session.rollback()
    session.close()

start = max(panel.prices.index[w[0]] for _, panel, _, w in runs.values())
labels = regimes.load()
for path, (spec, panel, weights, (lo, hi)) in runs.items():
    lo = int(panel.prices.index.searchsorted(start))
    part = _slice_rows(panel, lo, hi)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    result = run_backtest(part, weights.iloc[lo:hi + 1], costs, part.funding)
    m = compute_metrics(result)
    r = regimes.breakdown(result.net_return, labels, periods_per_year(part.prices.index))
    print(f"{path}: {part.prices.index[0]:%Y-%m-%d}..{part.prices.index[-1]:%Y-%m-%d} "
          f"ann {m['ann_return_net']:+.1%} sharpe {m['sharpe_net']:.2f} dd {m['max_dd']:.1%} "
          + " ".join(f"{k}={r[f'regime_{k}_return']:+.1%}/{r[f'regime_{k}_max_dd']:.1%}"
                     for k in ("bull", "flat", "bear")))
