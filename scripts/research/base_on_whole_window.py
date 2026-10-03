"""Base Bv2 on the experiments' own windows, whole window judged (as the
experiments were), inside a session that is rolled back: nothing recorded."""
import datetime as dt
import sys
from pathlib import Path

from qlab.pipeline.evaluate import evaluate_spec
from qlab.pipeline.spec import load_spec
from qlab.registry.db import get_sessionmaker
from qlab.rules.loader import load_latest

KEYS = ("ann_return_net", "sharpe_net", "max_dd", "selection_days", "ann_return_net_ex_best_1pct",
        "regime_bull_return", "regime_flat_return", "regime_bear_return",
        "regime_bull_days", "regime_flat_days", "regime_bear_days", "book_coverage")
for path in sys.argv[1:]:
    spec = load_spec(Path(path))
    spec = spec.model_copy(update={"params_fixed_at": dt.date(2026, 10, 3)})
    session = get_sessionmaker()()
    try:
        ev = evaluate_spec(spec, session=session, ruleset=load_latest(),
                           deployable_capital_usd=3000, update_idea_status=False,
                           check_lookahead=False, check_sources=False)
        m = ev.metrics or {}
        print(path, {k: round(m[k], 4) for k in KEYS if k in m})
    finally:
        session.rollback()
        session.close()
