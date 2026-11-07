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


STATE_KEYS = ("ba", "detected_share", "delay_median_days", "fall_lost_mean",
              "false_per_coin_year", "wrong_share", "fall_caught_share", "rise_lost_share",
              "switches_per_coin_year")


def detect_report(work: Path) -> None:
    with open(work / "detect.pkl", "rb") as fh:
        d = pickle.load(fh)
    res = d["results"]
    rep: dict = {"seconds": d["seconds"], "states": {}, "switch": {}, "t0_auc": {},
                 "train_vs_oos": {}, "common_from": {}}
    for grid in R.GRIDS:
        real = next(r for r in res if r["kind"] == "states" and r["grid"] == grid
                    and r["seed"] is None)
        rw = [r for r in res if r["kind"] == "states" and r["grid"] == grid
              and r["seed"] is not None]
        rep["common_from"][grid] = real["common_from"]
        rules = [n for n in real["states"] if n.startswith("rule")]
        best_rule = {p: max(real["states"][n][p]["ba"] for n in rules) for p in ("first", "last")}
        table = {}
        for name, per in real["states"].items():
            row = {}
            for p in ("first", "last"):
                ba_rw = np.array([r["states"][name][p]["ba"] for r in rw])
                row[p] = {k: per[p][k] for k in STATE_KEYS}
                row[p]["rw_ba_mean"] = float(ba_rw.mean())
                row[p]["rw_ba_p95"] = float(np.quantile(ba_rw, 0.95))
                row[p]["rw_draws"] = len(ba_rw)
                for k in ("delay_median_days", "fall_lost_mean", "false_per_coin_year",
                          "rise_lost_share"):
                    row[p][f"rw_{k}_mean"] = float(np.nanmean([r["states"][name][p][k]
                                                               for r in rw]))
                row[p]["beats_rules"] = bool(per[p]["ba"] > best_rule[p])
                row[p]["beats_rw"] = bool(per[p]["ba"] > row[p]["rw_ba_p95"])
            row["passes"] = all(row[p]["beats_rules"] and row[p]["beats_rw"]
                                for p in ("first", "last")) and not name.startswith("rule")
            table[name] = row
        rep["states"][grid] = {"best_rule_ba": best_rule, "table": table}
        for model in ("logistic", "gbm"):
            k = f"{model} T0 auc"
            rep["t0_auc"][f"{grid} {model}"] = {
                "real": real[k],
                "rw_auc_last": [r[k]["last"]["auc"] for r in rw],
                "rw_auc_first": [r[k]["first"]["auc"] for r in rw]}
            recs = real[f"{model} T0 record"]
            rep["train_vs_oos"][f"{grid} {model} T0"] = {
                "auc_in_mean": float(np.nanmean([x["auc_in"] for x in recs])),
                "auc_oos_first": real[k]["first"]["auc"], "auc_oos_last": real[k]["last"]["auc"]}
        # switch targets
        for tgt in R.SUP_TARGETS[1:]:
            geo = next(r for r in res if r["kind"] == f"{tgt}|geometry" and r["grid"] == grid
                       and r["seed"] is None)
            for model in ("logistic", "gbm", "geometry"):
                kind = f"{tgt}|{model}"
                real_s = next(r for r in res if r["kind"] == kind and r["grid"] == grid
                              and r["seed"] is None)
                rw_s = [r for r in res if r["kind"] == kind and r["grid"] == grid
                        and r["seed"] is not None]
                row = {}
                for p in ("first", "last"):
                    a = real_s["scores"][p]
                    rw_auc = np.array([r["scores"][p]["auc"] for r in rw_s])
                    row[p] = {**a, "rw_auc_mean": float(np.nanmean(rw_auc)),
                              "rw_auc_p95": float(np.nanquantile(rw_auc, 0.95)),
                              "geometry_auc": geo["scores"][p]["auc"]}
                    row[p]["passes"] = bool(a["auc"] > a["null_p95"]
                                            and a["auc"] > row[p]["rw_auc_p95"]
                                            and (model == "geometry"
                                                 or a["auc"] > geo["scores"][p]["auc"]))
                row["passes"] = row["first"]["passes"] and row["last"]["passes"]
                row["auc_in_mean"] = float(np.nanmean([x["auc_in"] for x in real_s["record"]]))
                rep["switch"][f"{grid} {kind}"] = row
    (work / "detect-report.json").write_text(json.dumps(rep, indent=1, default=float))
    for grid in R.GRIDS:
        s = rep["states"][grid]
        print(f"== states {grid}; best rule BA first {s['best_rule_ba']['first']:.3f} last "
              f"{s['best_rule_ba']['last']:.3f}; common from {rep['common_from'][grid]}")
        print("  name                    | BA f/l (RW mean, p95) | detected, delay d, lost, "
              "false/yr, rise lost (last) | pass")
        for name, row in s["table"].items():
            f, la = row["first"], row["last"]
            print(f"  {name:22s} | {f['ba']:.3f} ({f['rw_ba_mean']:.3f},{f['rw_ba_p95']:.3f}) "
                  f"{la['ba']:.3f} ({la['rw_ba_mean']:.3f},{la['rw_ba_p95']:.3f}) | "
                  f"{la['detected_share']:.2f} {la['delay_median_days']:.1f} "
                  f"{la['fall_lost_mean']:.2f} {la['false_per_coin_year']:.1f} "
                  f"{la['rise_lost_share']:.2f} | first: delay {f['delay_median_days']:.1f} "
                  f"lost {f['fall_lost_mean']:.2f} false {f['false_per_coin_year']:.1f} "
                  f"rise {f['rise_lost_share']:.2f} | RW last delay "
                  f"{la['rw_delay_median_days_mean']:.1f} lost {la['rw_fall_lost_mean_mean']:.2f}"
                  f" false {la['rw_false_per_coin_year_mean']:.1f} | {row['passes']}")
    print("== switch targets: AUC first/last (null p95, RW p95, geometry) precision@10 base")
    for k, row in rep["switch"].items():
        f, la = row["first"], row["last"]
        print(f"  {k:24s} {f['auc']:.3f} ({f['null_p95']:.3f},{f['rw_auc_p95']:.3f},"
              f"{f['geometry_auc']:.3f}) | {la['auc']:.3f} ({la['null_p95']:.3f},"
              f"{la['rw_auc_p95']:.3f},{la['geometry_auc']:.3f}) p10 {la['precision_top10']:.3f}"
              f" base {la['base_rate']:.3f} in-sample {row['auc_in_mean']:.3f} | {row['passes']}")
    print("== T0 AUC", json.dumps({k: {p: round(v["real"][p]["auc"], 3) for p in ("first", "last")}
                                   for k, v in rep["t0_auc"].items()}))
    print("== train vs oos", json.dumps(rep["train_vs_oos"]))


if __name__ == "__main__":
    main()
