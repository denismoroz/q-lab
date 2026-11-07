"""Tables of the regime-change study from the runner's outputs
(docs/research/regime-change-protocol-2026-11-07.md).

    uv run python scripts/research/regime_change_report.py oracle --work DIR
    uv run python scripts/research/regime_change_report.py detect --work DIR
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import regime_change as R  # noqa: E402

from qlab import regime_change as rc  # noqa: E402


def _gain(res, book, name, part, key="cap_ann"):
    return res[f"{book} {name}"][f"{key}_{part}"] - res[f"{book} production"][f"{key}_{part}"]


def oracle_report(work: Path) -> dict:
    o = json.loads((work / "oracle.json").read_text())
    res = o["results"]
    out = {"label": o["label"], "production": {}, "exact": {}, "degraded": {}}
    for cap in R.CAPITALS:
        books = [f"{c}@{cap:g}" for c in R.COINS]
        out["production"][f"{cap:g}"] = {b: res[f"{b} production"] for b in books}
        ex = {}
        for v in R.EXACT:
            row = {}
            for part in ("first", "last"):
                g = [_gain(res, b, v, part) for b in books]
                dd = [_gain(res, b, v, part, "cap_max_dd") for b in books]
                row[part] = {"gain_mean": float(np.mean(g)), "gain_by_coin": dict(zip(R.COINS, g)),
                             "dd_change_mean": float(np.mean(dd)),
                             "ons_per_year": float(np.mean([res[f"{b} {v}"][f"ons_per_year_{part}"]
                                                           for b in books]))}
            ex[v] = row
        out["exact"][f"{cap:g}"] = ex
        dg = {}
        for d in R.DELAYS:
            for f in R.FALSE_RATES:
                row = {}
                for part in ("first", "last"):
                    per_seed = []
                    dds = []
                    for s in range(R.SEEDS if f > 0 else 1):
                        name = f"delay {d} false {f} seed {s}"
                        per_seed.append(np.mean([_gain(res, b, name, part) for b in books]))
                        dds.append(np.mean([_gain(res, b, name, part, "cap_max_dd")
                                            for b in books]))
                    row[part] = {"gain_mean": float(np.mean(per_seed)),
                                 "gain_min": float(np.min(per_seed)),
                                 "gain_max": float(np.max(per_seed)),
                                 "dd_change_mean": float(np.mean(dds))}
                dg[f"d{d} f{f}"] = row
        out["degraded"][f"{cap:g}"] = dg
    # where production stands on the yardstick (its open hedge at the daily close)
    frames = R.load_frames(work)
    with open(work / "prod_hedged.pkl", "rb") as fh:
        ph = pickle.load(fh)
    gs = R.grid_series(frames, 24)
    labs = {c: R.labels_for(g, R.X_MAIN) for c, g in gs.items()}
    prod_state = {}
    for c, g in gs.items():
        h = np.asarray(ph[f"{c}@1000"])
        prod_state[c] = h[g["rows"]]
    per = {}
    for part in ("first", "last"):
        parts = []
        for c, g in gs.items():
            mask = R.periods_of(g["time"])[part] & ~np.isnan(labs[c]["state"])
            parts.append(rc.yardstick(g["logc"], labs[c]["state"], prod_state[c], labs[c]["legs"],
                                      mask, 24))
        per[part] = rc.pool(parts)
    out["production_on_yardstick"] = per
    # the degraded oracles on the same yardstick (the wishes re-made with the runner's seeds)
    W = R.oracle_wishes(frames)
    deg = {}
    for d in R.DELAYS:
        for f in R.FALSE_RATES:
            rows = {"first": [], "last": []}
            for s in range(R.SEEDS if f > 0 else 1):
                parts = {"first": [], "last": []}
                for c, g in gs.items():
                    rng = np.random.default_rng(100_000 + 1000 * R.COINS.index(c) + 100 * d
                                                + 10 * f + s)
                    w = rc.flip_noise(rc.delay_wish(W[c]["R1 25% 4h"], 24 * d), f, rng)
                    st = w[g["rows"]]
                    for part in parts:
                        mask = R.periods_of(g["time"])[part] & ~np.isnan(labs[c]["state"])
                        parts[part].append(rc.yardstick(g["logc"], labs[c]["state"], st,
                                                        labs[c]["legs"], mask, 24))
                for part in parts:
                    rows[part].append(rc.pool(parts[part]))
            deg[f"d{d} f{f}"] = {part: {k: float(np.nanmean([r[k] for r in rows[part]]))
                                        for k in rows[part][0]} for part in rows}
    out["degraded_on_yardstick"] = deg
    out["runs"] = o["runs"]
    out["seconds"] = o["seconds"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["oracle", "detect"])
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    if a.phase == "oracle":
        rep = oracle_report(a.work)
        (a.work / "oracle-report.json").write_text(json.dumps(rep, indent=1, default=float))
        for cap in ("1000", "64.25"):
            print("== exact, $", cap)
            for v, row in rep["exact"][cap].items():
                print(f"  {v:28s} first {row['first']['gain_mean']:+6.1f} (dd {row['first']['dd_change_mean']:+5.1f})"
                      f"  last {row['last']['gain_mean']:+6.1f} (dd {row['last']['dd_change_mean']:+5.1f})"
                      f"  ons/yr {row['last']['ons_per_year']:.1f}  by coin last "
                      + " ".join(f"{k} {x:+.0f}" for k, x in row['last']['gain_by_coin'].items()))
            print("== degraded, $", cap)
            for k, row in rep["degraded"][cap].items():
                print(f"  {k:8s} first {row['first']['gain_mean']:+6.1f} [{row['first']['gain_min']:+.1f},{row['first']['gain_max']:+.1f}]"
                      f"  last {row['last']['gain_mean']:+6.1f} [{row['last']['gain_min']:+.1f},{row['last']['gain_max']:+.1f}]"
                      f"  dd last {row['last']['dd_change_mean']:+5.1f}")
        for cap in ("1000", "64.25"):
            for b, m in rep["production"][cap].items():
                print("prod", b, {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()})
        keys = ("ba", "delay_median_days", "fall_lost_mean", "false_per_coin_year",
                "fall_caught_share", "rise_lost_share", "switches_per_coin_year")
        print("yardstick (last third):", keys)
        print("  production", [round(rep["production_on_yardstick"]["last"][k], 2) for k in keys])
        for k, row in rep["degraded_on_yardstick"].items():
            print(f"  {k:8s}", [round(row["last"][x], 2) for x in keys])
    else:
        detect_report(a.work)


def detect_report(work: Path) -> None:
    raise SystemExit("filled in after the detection run")


if __name__ == "__main__":
    main()
