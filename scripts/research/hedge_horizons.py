"""Over what span is Bv2's hedge outcome predictable? (docs/TASKS.md T39;
owner, 2026-10-04: «если взять больше чем 14 дней, или взять часовые свечи а
не дневные»).

Development only: level 2 walked forward on 2020-08 .. 2023-12 for each
horizon, with and without level 1's probabilities, scored against the same
labels as the book's own 14/30-day rule and the two constant answers. The
holdout (2024-01 on) is scored once for the variant chosen here
(`--holdout HOURS [--no-level1]`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from qlab import meta_hedge as mh
from qlab.pipeline.evaluate import resolve_panel
from qlab.pipeline.spec import load_spec
from qlab.regime_detect import load_predictions
from qlab.registry.db import get_sessionmaker

DEV = (pd.Timestamp("2020-08-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"))


def main() -> None:
    spec = load_spec(Path("specs/bv2-binance.yaml"))
    session = get_sessionmaker()()
    try:
        panel = resolve_panel(session, spec)
    finally:
        session.rollback()
        session.close()
    level1 = load_predictions("btc-returns-logistic")
    cost = mh.round_trip_cost(dict(spec.params))
    coins = list(spec.params["coins"])
    feats_all = {c: mh.coin_features(panel.prices[c], panel.funding[c], level1) for c in coins}
    own = {}
    for c in coins:
        perp = panel.prices[c]
        own[c] = ~((perp / perp.shift(14 * 24) - 1 > 0) & (perp / perp.shift(30 * 24) - 1 > 0))

    def score(horizon: int, use_level1: bool, window) -> dict[str, float]:
        feats = feats_all if use_level1 else {
            c: f.drop(columns=["p_bull", "p_bear"]) for c, f in feats_all.items()}
        labels = {c: mh.hedge_pays(panel.prices[c], panel.funding[c], cost, horizon)
                  for c in coins}
        dec = mh.walk_forward(feats, labels, window[0], window[1], horizon)
        rows = []
        for c in coins:
            both = pd.DataFrame({"m": dec[c], "o": own[c], "y": labels[c]}).dropna()
            both = both[(both.index >= window[0]) & (both.index < window[1])]
            rows.append(both)
        b = pd.concat(rows)
        y = b["y"] == 1.0
        return {"pays": y.mean(), "model": (b["m"].astype(bool) == y).mean(),
                "book": (b["o"].astype(bool) == y).mean(),
                "always": y.mean(), "never": 1 - y.mean(), "hedged": b["m"].astype(bool).mean()}

    if "--holdout" in sys.argv:
        horizon = int(sys.argv[sys.argv.index("--holdout") + 1])
        s = score(horizon, "--no-level1" not in sys.argv, HOLDOUT)
        print(f"holdout {horizon // 24}d: " + " ".join(f"{k} {v:.3f}" for k, v in s.items()))
        return
    for days in (7, 14, 30, 60):
        for use in (True, False):
            s = score(days * 24, use, DEV)
            tag = "with level 1" if use else "without"
            print(f"{days:>2}d {tag:<12} " + " ".join(f"{k} {v:.3f}" for k, v in s.items()),
                  flush=True)


if __name__ == "__main__":
    main()
