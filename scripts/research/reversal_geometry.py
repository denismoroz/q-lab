"""Diagnostic added after the first run (not declared before it): is the reversal detector's AUC
just the geometry of the event — how far the event bar overshot the x·σ step? Compares, on the
large bars, the full model with a model on the overshoot alone and a model without the columns
that carry it. Writes DIR/reversal-geometry.json.

    uv run python scripts/research/reversal_geometry.py --work DIR
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

KEYS = ("0.5_1.5", "1_5")
GEOMETRY = ("step_sigma", "speed", "ret_1", "ret_2")


def _job(args):
    work, key, side, variant = args
    import reversal_run as rr

    from qlab import reversal as rv

    names = list(rv.FEATURE_NAMES) + ["btc_ret_4", "btc_ret_16"]
    geo = [names.index(g) for g in GEOMETRY]
    if variant == "overshoot only":
        cols = [names.index("step_sigma")]
        model = "logit"
    else:
        cols = [j for j in range(len(names)) if j not in geo]
        model = "gbm"
    cols = cols + list(range(len(names), len(names) + len(rr.COINS)))  # keep the coin id

    def sub(A):
        return {**A, "X": A["X"][:, cols]}

    D = sub(rr.load(work, "15m", key, side))
    apply = {tf: sub(rr.load(work, tf, key, side)) for tf in rr.LARGE}
    if variant == "overshoot only":
        preds = {tf: A["X"][:, 0].astype(float) for tf, A in apply.items()}  # a monotone rule
    else:
        preds, _ = rr.fit_predict(model, D, apply)
    return key, side, variant, {tf: rr.measure(A, preds[tf]) for tf, A in apply.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    jobs = [(a.work, k, s, v) for k in KEYS for s in (1, -1)
            for v in ("overshoot only", "without the geometry columns")]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        got = pool.map(_job, jobs)
    out = {f"{k} {'top' if s == 1 else 'bottom'} {v}": r for k, s, v, r in got}
    (a.work / "reversal-geometry.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {tf: (round(x["first"]["auc"], 3), round(x["last"]["auc"], 3))
                  for tf, x in v.items()})


if __name__ == "__main__":
    main()
