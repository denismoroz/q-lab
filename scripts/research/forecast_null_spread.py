"""Forecast pilot 2, diagnostic after the fact: the spread of the primary null over repeated
draws (one draw's bootstrap interval holds the fitted null model fixed, so it does not show how
far another draw would land).

    uv run python scripts/research/forecast_null_spread.py --work DIR --variant b --draws 20

Writes DIR/forecast2-null-spread-<variant>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import forecast_pilot as fp  # noqa: E402
import forecast_pilot2 as fp2  # noqa: E402

from qlab import forecast as fc  # noqa: E402
from qlab import forecast_inputs as fi  # noqa: E402

TARGETS = [("reg", 4, None), ("reg", 24, None), ("cls", 1, "fall"), ("cls", 1, "rise"),
           ("cls", 4, "fall"), ("cls", 4, "rise")]


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    ap.add_argument("--variant", default="b")
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    V = fp2.build(a.work, a.minutes, a.variant)
    tau, X, d = V["tau"], V["X"], V["d"]
    retrains = pd.date_range(fp.START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
    out: dict = {"variant": a.variant, "draws": a.draws}
    for t in TARGETS:
        kind, h, s = t
        y = V["targets"][h] if kind == "reg" else V["labels"][(h, s)]
        vals = {"first": [], "last": []}
        for k in range(a.draws):
            rng = np.random.default_rng(1000 + k)
            p = np.full(len(tau), np.nan)
            for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
                m = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= q0.value) & ~np.isnan(y) & (
                    tau >= d["start"])
                sel = (tau >= q0.value) & (tau < q1.value)
                ys = fi.shuffled_labels(y, m, tau, "other day", rng)
                if kind == "reg":
                    g = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                                      learning_rate=0.05,
                                                      min_samples_leaf=1000, random_state=0)
                    p[sel] = g.fit(X[m], ys[m]).predict(X[sel])
                else:
                    g = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                       learning_rate=0.05,
                                                       min_samples_leaf=1000, random_state=0)
                    p[sel] = g.fit(X[m], ys[m]).predict_proba(X[sel])[:, 1]
            for part, mask in fp2.parts(tau, h).items():
                ok = mask & ~np.isnan(p) & ~np.isnan(y)
                vals[part].append(fc.pearson(p[ok], y[ok]) if kind == "reg"
                                  else fc.auc(p[ok], y[ok]))
        name = f"z {h}h gbm" if kind == "reg" else f"{h}h {s} gbm"
        out[name] = {part: {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v)),
                            "q025": float(np.quantile(v, 0.025)),
                            "q975": float(np.quantile(v, 0.975))} for part, v in vals.items()}
        print(name, {p_: (round(x["mean"], 3), round(x["sd"], 3)) for p_, x in out[name].items()},
              round(time.time() - t0), "s", flush=True)
    out["trials"] = len(TARGETS) * a.draws
    out["seconds"] = round(time.time() - t0)
    (a.work / f"forecast2-null-spread-{a.variant}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
